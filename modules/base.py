from functools import partial
from typing import Dict, Any, Optional, Type

import pytorch_lightning as pl
import torch
from torch import nn

class BaseModule(pl.LightningModule):
    def __init__(
        self, 
        backbone: nn.Module, 
        optimizer: Type, 
        optimizer_kwargs: Dict[str, Any], 
        scheduler: Type = None, 
        scheduler_kwargs: Dict[str, Any] = None,
        projection_head: Optional[nn.Module] = None,
        prediction_head: Optional[nn.Module] = None,
        prototypes: Optional[nn.Module] = None,
        linear_head: Optional[nn.Module] = None,
    ):
        super(BaseModule, self).__init__()

        self.backbone = backbone
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.prototypes = prototypes
        self.linear = linear_head
        
        self.optimizer = partial(optimizer, **optimizer_kwargs)
        self.save_hyperparameters(optimizer_kwargs)

        if scheduler and scheduler_kwargs:
            self.scheduler = partial(scheduler, **scheduler_kwargs)
            self.save_hyperparameters(scheduler_kwargs)

    def configure_optimizers(self):
        optim = self.optimizer(
            filter(lambda x: x.requires_grad, self.parameters()),
        )
        if hasattr(self, "scheduler"):
            scheduler = self.scheduler(optim)
            scheduler = {
                "scheduler": scheduler,
                "interval": "step",
            }
            return [optim], [scheduler]
        
        return optim
    
    def validation_step(self, batch, batch_index):
        with torch.no_grad():
            valid_loss = self.training_step(batch, batch_index)
        self.log("valid-ssl-loss", valid_loss.cpu().item(), sync_dist=True)