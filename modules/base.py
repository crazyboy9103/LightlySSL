from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn

class BaseModule(pl.LightningModule):
    def __init__(
        self, 
        backbone: nn.Module, 
        batch_size_per_device: int, 
        projection_head: Optional[nn.Module] = None,
        prediction_head: Optional[nn.Module] = None,
        prototypes: Optional[nn.Module] = None,
        linear_head: Optional[nn.Module] = None,
    ):
        super(BaseModule, self).__init__()

        self.backbone = backbone
        self.batch_size_per_device = batch_size_per_device
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.prototypes = prototypes
        self.linear_head = linear_head
        
        self.is_distributed = torch.cuda.device_count() > 1

    def validation_step(self, batch, batch_index):
        x, y = batch
        z = self.backbone(x).flatten(start_dim=1)
        _, loss_dict = self.linear_head.validation_step((z, y), batch_index)
        self.log_dict(loss_dict, sync_dist=self.is_distributed)
    
    def on_validation_epoch_end(self):
        metrics = self.linear_head.on_validation_epoch_end()
        self.log_dict(metrics, sync_dist=self.is_distributed)
    
    def on_train_epoch_end(self):
        metrics = self.linear_head.on_train_epoch_end()
        self.log_dict(metrics, sync_dist=self.is_distributed)
        