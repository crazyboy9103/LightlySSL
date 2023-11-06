import copy 

import torch
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.utils.scheduler import cosine_schedule

from .base import BaseModule

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
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    
    def training_step(self, batch, batch_index):
        # momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        momentum = 0.996
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1) = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train-ssl-loss", loss, sync_dist=self.is_distributed)
        return loss
