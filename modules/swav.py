import torch
from torch.nn import functional as F
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes

from .base import BaseModule

class SwAV(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2), 
        scheduler = None, 
        scheduler_kwargs = None,
        projection_head_kwargs = dict(hidden_dim=512, output_dim=128),
        prototype_kwargs = dict(n_prototypes=512),
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            SwaVProjectionHead(
                backbone.output_dim, 
                projection_head_kwargs["hidden_dim"], 
                projection_head_kwargs["output_dim"]
            ),
            None,
            SwaVPrototypes(
                projection_head_kwargs["output_dim"], 
                n_prototypes=prototype_kwargs["n_prototypes"]
            )
        )
        
        self.criterion = SwaVLoss()
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(prototype_kwargs)
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = F.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_index):
        self.prototypes.normalize()
        views = batch[0]
        multi_crop_features = [self.forward(view.to(self.device)) for view in views]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log("train-loss", loss)
        return loss