import torch
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

from .base import BaseModule
from .eval import OnlineClassifier

class SimCLR(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2), 
        scheduler = None, 
        scheduler_kwargs = None,
        projection_head_kwargs = dict(hidden_dim=2048, output_dim=2048),
        linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1, k=15),
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            projection_head=SimCLRProjectionHead(
                input_dim=backbone.output_dim, 
                hidden_dim=projection_head_kwargs["hidden_dim"], 
                output_dim=projection_head_kwargs["output_dim"]
            ),
            linear_head=OnlineClassifier(
                input_dim=backbone.output_dim, 
                num_classes=linear_head_kwargs["num_classes"],
                label_smoothing=linear_head_kwargs["label_smoothing"],
                k=linear_head_kwargs["k"]
            )
        )
        self.criterion = NTXentLoss(gather_distributed=self.is_distributed)
        
        self.save_hyperparameters(projection_head_kwargs)
        
    def forward(self, x):
        z = self.backbone(x).flatten(start_dim=1)
        p = self.projection_head(z)
        return z, p

    def training_step(self, batch, batch_index):
        (x0, x1), y = batch
        z0, p0 = self.forward(x0)
        _, p1 = self.forward(x1)
        loss = self.criterion(p0, p1)
        loss_dict = {"train-ssl-loss": loss}
        
        cls_loss, cls_loss_dict = self.linear_head.training_step((z0.detach(), y), batch_index)
        loss_dict.update(cls_loss_dict)
        self.log_dict(loss_dict, sync_dist=self.is_distributed)
        return loss + cls_loss