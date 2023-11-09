import copy 

import torch
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.utils.scheduler import cosine_schedule

from .base import BaseModule
from .eval import OnlineClassifier

class BYOL(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2), 
        scheduler = None, 
        scheduler_kwargs = None,
        projection_head_kwargs = dict(hidden_dim=1024, output_dim=256),
        prediction_head_kwargs = dict(hidden_dim=1024, output_dim=256),
        linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1, k=15),
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            projection_head=BYOLProjectionHead(
                input_dim=backbone.output_dim, 
                hidden_dim=projection_head_kwargs["hidden_dim"], 
                output_dim=projection_head_kwargs["output_dim"]
            ),
            prediction_head=BYOLPredictionHead(
                input_dim=projection_head_kwargs["output_dim"],  
                hidden_dim=prediction_head_kwargs["hidden_dim"],
                output_dim=prediction_head_kwargs["output_dim"]
            ),
            linear_head=OnlineClassifier(
                input_dim=backbone.output_dim, 
                num_classes=linear_head_kwargs["num_classes"],
                label_smoothing=linear_head_kwargs["label_smoothing"],
                k=linear_head_kwargs["k"]
            )
        )
        
        self.backbone_momentum = copy.deepcopy(backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        
        self.criterion = NegativeCosineSimilarity()
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(prediction_head_kwargs)
        
    def forward(self, x):
        z = self.backbone(x).flatten(start_dim=1)
        y = self.projection_head(z)
        p = self.prediction_head(y)
        return z, p

    def forward_momentum(self, x):
        z = self.backbone_momentum(x).flatten(start_dim=1)
        y = self.projection_head_momentum(z)
        y = y.detach()
        return y
    
    def training_step(self, batch, batch_index):
        # momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        momentum = 0.996
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1), y = batch
        z0, p0 = self.forward(x0)
        y0 = self.forward_momentum(x0)
        _, p1 = self.forward(x1)
        y1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, y1) + self.criterion(p1, y0))
        
        cls_loss, cls_loss_dict = self.linear_head.training_step((z0.detach(), y), batch_index)
        loss_dict = {"train-ssl-loss": loss}
        loss_dict.update(cls_loss_dict)
        self.log_dict(loss_dict, sync_dist=self.is_distributed)
        return loss + cls_loss

