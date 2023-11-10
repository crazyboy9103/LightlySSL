import copy 

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import (
    deactivate_requires_grad, 
    update_momentum, 
    get_weight_decay_parameters
)
from lightly.utils.scheduler import cosine_schedule
from lightly.utils.scheduler import CosineWarmupScheduler

from torch.optim import SGD

from .base import BaseModule
from .eval import OnlineClassifier

class DINO(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=512, bottleneck_dim=64, output_dim=2048),
        loss_kwargs = dict(warmup_teacher_temp_epochs=5),
        linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1, k=15),
    ):
        super().__init__(
            backbone, 
            batch_size_per_device,
            projection_head=DINOProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            linear_head=OnlineClassifier(
                input_dim=backbone.output_dim, 
                **linear_head_kwargs
            )
        )
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(loss_kwargs)
        self.save_hyperparameters(linear_head_kwargs)
        
        # teacher model dont freeze last layer
        projection_head_kwargs.pop("freeze_last_layer", None)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_projection_head = DINOProjectionHead(
            input_dim=backbone.output_dim, 
            **projection_head_kwargs
        )
        
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_projection_head)

        self.criterion = DINOLoss(
            output_dim=projection_head_kwargs["output_dim"],
            **loss_kwargs
        )
        
        
    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return y, z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_projection_head(y)
        return y, z

    def training_step(self, batch, batch_index):
        momentum = cosine_schedule(
            self.trainer.global_step, 
            self.trainer.estimated_stepping_batches, 
            start_value=0.996, 
            end_value=1
        )
        update_momentum(self.backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)
        views, targets = batch
        global_views = views[:2]
        local_views = views[2:]
        
        teacher_output = [self.forward_teacher(view) for view in global_views]
        teacher_features, teacher_projections = zip(*teacher_output)
        
        student_output = [self.forward(view) for view in views]
        _, student_projections = zip(*student_output)        
        
        loss = self.criterion(teacher_projections, student_projections, epoch=self.current_epoch)
        
        loss_dict = {"train-ssl-loss": loss}
        
        cls_loss, cls_loss_dict = self.linear_head.training_step((teacher_features[0].detach(), targets), batch_index)
        loss_dict.update(cls_loss_dict)
        
        self.log_dict(loss_dict, sync_dist=self.is_distributed)           
        return loss + cls_loss

    def on_after_backward(self):
        self.projection_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)
        
    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        # For ResNet50 we use SGD instead of AdamW/LARS as recommended by the authors:
        # https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
        optimizer = SGD(
            [
                {"name": "dino", "params": params},
                {
                    "name": "dino_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.linear_head.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.03 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]