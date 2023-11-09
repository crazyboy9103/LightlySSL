import torch
from torch.nn import functional as F
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes

from .base import BaseModule
from .eval import OnlineClassifier
# TODO add queue
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
        linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1, k=15),
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            projection_head=SwaVProjectionHead(
                input_dim=backbone.output_dim, 
                hidden_dim=projection_head_kwargs["hidden_dim"], 
                output_dim=projection_head_kwargs["output_dim"]
            ),
            prototypes=SwaVPrototypes(
                input_dim=projection_head_kwargs["output_dim"], 
                n_prototypes=prototype_kwargs["n_prototypes"]
            ),
            linear_head=OnlineClassifier(
                input_dim=backbone.output_dim, 
                num_classes=linear_head_kwargs["num_classes"],
                label_smoothing=linear_head_kwargs["label_smoothing"],
                k=linear_head_kwargs["k"]
            )
        )
        
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=self.is_distributed)
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(prototype_kwargs)
        
    def forward(self, x):
        z = self.backbone(x).flatten(start_dim=1)
        p = self.projection_head(z)
        p = F.normalize(p, dim=1, p=2)
        p = self.prototypes(p)
        return z, p

    def training_step(self, batch, batch_index):
        self.prototypes.normalize()
        views, y = batch
        outputs = [self.forward(view) for view in views]
        features, prototypes = zip(*outputs)
        high_resolution = prototypes[:2]
        low_resolution = prototypes[2:]
        loss = self.criterion(high_resolution, low_resolution)
        loss_dict = {"train-ssl-loss": loss}
        
        cls_loss, cls_loss_dict = self.linear_head.training_step((features[0].detach(), y), batch_index)
        loss_dict.update(cls_loss_dict)        
        self.log_dict(loss_dict, sync_dist=self.is_distributed)        
        return loss + cls_loss