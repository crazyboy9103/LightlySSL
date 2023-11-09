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
from .eval import OnlineClassifier
# TODO check https://github.com/facebookresearch/moco for correct implementation

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
        linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1, k=15),
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
            ),
            linear_head=OnlineClassifier(
                input_dim=backbone.output_dim, 
                num_classes=linear_head_kwargs["num_classes"],
                label_smoothing=linear_head_kwargs["label_smoothing"],
                k=linear_head_kwargs["k"]
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
        z = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(z)
        return z, query

    def forward_momentum(self, x):
        z = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(z).detach()
        return z, key    
    
    def training_step(self, batch, batch_index):
        momentum = cosine_schedule(
            self.trainer.global_step, 
            self.trainer.estimated_stepping_batches, 
            start_value=0.996, 
            end_value=1
        )
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x_query, x_key), y = batch
        z_query, query = self.forward(x_query)
        with torch.no_grad():
            x_key, shuffle = self._batch_shuffle(x_key)
            _, key = self.forward_momentum(x_key)
            key = self._batch_unshuffle(key, shuffle)
        
        loss = self.criterion(query, key)
        loss_dict = {"train-ssl-loss": loss}
        
        cls_loss, cls_loss_dict = self.linear_head.training_step((z_query.detach(), y), batch_index)
        loss_dict.update(cls_loss_dict)
        self.log_dict(loss_dict, sync_dist=self.is_distributed)        
        return loss + cls_loss
    
    @torch.no_grad()
    def _batch_shuffle(self, batch: torch.Tensor):
        """Returns the shuffled batch and the indices to undo."""
        batch_size = batch.shape[0]
        shuffle = torch.randperm(batch_size, device=batch.device)
        return batch[shuffle], shuffle
    
    @torch.no_grad()
    def _batch_unshuffle(self, batch: torch.Tensor, shuffle: torch.Tensor):
        """Returns the unshuffled batch."""
        unshuffle = torch.argsort(shuffle)
        return batch[unshuffle]