import torch
from lightly.loss.vicreg_loss import VICRegLoss
## The projection head is the same as the Barlow Twins one
from lightly.models.modules.heads import VICRegProjectionHead

from .base import BaseModule

class VICReg(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2), 
        scheduler = None, 
        scheduler_kwargs = None,
        projection_head_kwargs = dict(hidden_dim=1024, output_dim=256, num_layers=2),
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            projection_head=VICRegProjectionHead(
                input_dim=backbone.output_dim, 
                hidden_dim=projection_head_kwargs["hidden_dim"], 
                output_dim=projection_head_kwargs["output_dim"],
                num_layers=projection_head_kwargs["num_layers"]
            )
        )
        
        self.criterion = VICRegLoss(gather_distributed=self.is_distributed)
        
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
        self.log("train-ssl-loss", loss, sync_dist=self.is_distributed)
        return loss
