import copy

from lightly.loss import NTXentLoss
from lightly.models.utils import (
    deactivate_requires_grad,
    update_momentum,
)
from lightly.models.modules import MoCoProjectionHead
from lightly.utils.scheduler import cosine_schedule
import torch 

from .base import BaseModule

class MoCo(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2, momentum=0.9, weight_decay=5e-4), 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR, 
        scheduler_kwargs = dict(max_epochs=200),
        projection_head_kwargs = dict(hidden_dim=512, output_dim=128),
        loss_kwargs = dict(memory_bank_size=4096),
    ):
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            projection_head=MoCoProjectionHead(
                input_dim=backbone.output_dim, 
                hidden_dim=projection_head_kwargs["hidden_dim"], 
                output_dim=projection_head_kwargs["output_dim"]
            )
        )
        
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=loss_kwargs["temperature"],
            memory_bank_size=loss_kwargs["memory_bank_size"],
            gather_distributed=self.is_distributed
        )
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(loss_kwargs)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key    
    
    def training_step(self, batch, batch_index):
        # momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        momentum = 0.996
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        self.log("train-ssl-loss", loss, sync_dist=self.is_distributed)
        return loss