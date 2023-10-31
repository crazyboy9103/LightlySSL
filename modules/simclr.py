import torch
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

from .base import BaseModule

class SimCLR(BaseModule):
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
            projection_head=SimCLRProjectionHead(
                backbone.output_dim, 
                projection_head_kwargs["hidden_dim"], 
                projection_head_kwargs["output_dim"]
            )
        )
        self.criterion = NTXentLoss(gather_distributed=True if torch.cuda.device_count() > 1 else False)
        
        self.save_hyperparameters(projection_head_kwargs)
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train-ssl-loss", loss)
        return loss