import copy 

import torch
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from .base import BaseModule

class DINO(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2), 
        scheduler = None, 
        scheduler_kwargs = None,
        projection_head_kwargs = dict(hidden_dim=512, bottleneck_dim=64, output_dim=2048),
        loss_kwargs = dict(warmup_teacher_temp_epochs=5)
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            DINOProjectionHead(
                backbone.output_dim, 
                projection_head_kwargs["hidden_dim"], 
                projection_head_kwargs["bottleneck_dim"],
                projection_head_kwargs["output_dim"],
            )
        )
        
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_projection_head = DINOProjectionHead(
            backbone.output_dim, 
            projection_head_kwargs["hidden_dim"], 
            projection_head_kwargs["bottleneck_dim"],
            projection_head_kwargs["output_dim"],
        )
        
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_projection_head)
        
        self.criterion = DINOLoss(
            projection_head_kwargs["output_dim"],
            warmup_teacher_temp_epochs = loss_kwargs["warmup_teacher_temp_epochs"]
        )
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(loss_kwargs)
        
    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_projection_head(y)
        return z

    def training_step(self, batch, batch_index):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train-loss", loss)
        return loss

    def on_after_backward(self):
        self.projection_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)