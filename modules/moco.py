import copy

from lightly.loss import NTXentLoss
from lightly.models.utils import (
    deactivate_requires_grad,
    update_momentum,
    get_weight_decay_parameters
)
from lightly.models.modules import MoCoProjectionHead
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.scheduler import cosine_schedule

import torch 
from torch.optim import SGD

from .base import BaseModule
from .eval import OnlineClassifier
# TODO check https://github.com/facebookresearch/moco for correct implementation

class MoCo(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=512, output_dim=128),
        loss_kwargs = dict(memory_bank_size=4096),
        linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1, k=15),
    ):
        super().__init__(
            backbone, 
            batch_size_per_device,
            projection_head=MoCoProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            linear_head=OnlineClassifier(
                input_dim=backbone.output_dim, 
                **linear_head_kwargs
            )
        )
        
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            gather_distributed=self.is_distributed,
            **loss_kwargs
        )
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(loss_kwargs)
        self.save_hyperparameters(linear_head_kwargs)
        
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
    
    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        # For ResNet50 we use SGD instead of AdamW/LARS as recommended by the authors:
        # https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
        optimizer = SGD(
            [
                {"name": "dino", "params": params},
                {
                    "name": "dino_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.linear_head.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.03 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
    