import torch

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead

from .base import BaseModule

class BarlowTwins(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2), 
        scheduler = None, 
        scheduler_kwargs = None,
        projection_head_kwargs = dict(hidden_dim=2048, output_dim=2048),
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            BarlowTwinsProjectionHead(
                backbone.output_dim, 
                projection_head_kwargs["hidden_dim"],
                projection_head_kwargs["output_dim"]
            )
        )
        
        self.criterion = BarlowTwinsLoss()
        
        self.save_hyperparameters(projection_head_kwargs)
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train-loss", loss)
        return loss