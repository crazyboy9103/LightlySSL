import copy 

import torch
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from .base import BaseModule
from .eval import OnlineClassifier

class DINO(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2), 
        scheduler = None, 
        scheduler_kwargs = None,
        projection_head_kwargs = dict(hidden_dim=512, bottleneck_dim=64, output_dim=2048),
        loss_kwargs = dict(warmup_teacher_temp_epochs=5),
        linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1, k=15),
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            projection_head=DINOProjectionHead(
                input_dim=backbone.output_dim, 
                hidden_dim=projection_head_kwargs["hidden_dim"], 
                bottleneck_dim=projection_head_kwargs["bottleneck_dim"],
                output_dim=projection_head_kwargs["output_dim"],
                batch_norm=projection_head_kwargs["batch_norm"],
                freeze_last_layer=projection_head_kwargs["freeze_last_layer"],
                norm_last_layer=projection_head_kwargs["norm_last_layer"]
            ),
            linear_head=OnlineClassifier(
                input_dim=backbone.output_dim, 
                num_classes=linear_head_kwargs["num_classes"],
                label_smoothing=linear_head_kwargs["label_smoothing"],
                k=linear_head_kwargs["k"]
            )
        )
        
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_projection_head = DINOProjectionHead(
            input_dim=backbone.output_dim, 
            hidden_dim=projection_head_kwargs["hidden_dim"], 
            bottleneck_dim=projection_head_kwargs["bottleneck_dim"],
            output_dim=projection_head_kwargs["output_dim"],
            batch_norm=projection_head_kwargs["batch_norm"],
            freeze_last_layer=projection_head_kwargs["freeze_last_layer"],
            norm_last_layer=projection_head_kwargs["norm_last_layer"]
        )
        
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_projection_head)

        self.criterion = DINOLoss(
            output_dim=projection_head_kwargs["output_dim"],
            warmup_teacher_temp=loss_kwargs["warmup_teacher_temp"],
            teacher_temp=loss_kwargs["teacher_temp"],
            warmup_teacher_temp_epochs=loss_kwargs["warmup_teacher_temp_epochs"],
            student_temp=loss_kwargs["student_temp"],
            center_momentum=loss_kwargs["center_momentum"]
        )
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(loss_kwargs)
        
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