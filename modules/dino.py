import copy 
from typing import Union

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
from torch.optim.optimizer import Optimizer

from .base import BaseModule
from .eval import OnlineLinearClassifier

class DINO(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=512, bottleneck_dim=64, output_dim=2048),
        loss_kwargs = dict(warmup_teacher_temp_epochs=5),
        online_linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1),
    ):
        super().__init__(
            backbone, 
            batch_size_per_device,
            projection_head=DINOProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            online_linear_head=OnlineLinearClassifier(
                input_dim=backbone.output_dim, 
                **online_linear_head_kwargs
            )
        )
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(loss_kwargs)
        self.save_hyperparameters(online_linear_head_kwargs)
        
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
        y = self.backbone(x)
        z = self.projection_head(y)
        return y, z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        z = self.teacher_projection_head(y)
        return y, z

    def training_output(self, batch, batch_index):
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
        return {
            "loss": loss, 
            "embedding": teacher_features[0].detach(),
            "target": targets
        }

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
                {
                    "name": "dino_weight_decay", 
                    "params": params
                },
                {
                    "name": "dino_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_linear_head.parameters(),
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
    
    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Union[int, float, None] = None,
        gradient_clip_algorithm: Union[str, None] = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=gradient_clip_val or 2.0,
            gradient_clip_algorithm=gradient_clip_algorithm or "norm",
        )
        self.projection_head.cancel_last_layer_gradients(self.current_epoch)