import copy

from lightly.loss import NTXentLoss
from lightly.models.utils import (
    deactivate_requires_grad,
    update_momentum,
    get_weight_decay_parameters,
    batch_shuffle, 
    batch_unshuffle
)
from lightly.models.modules import MoCoProjectionHead
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.scheduler import cosine_schedule

import torch 
from torch.optim import SGD

from .base import BaseModule
from .eval import OnlineLinearClassifier, kNNClassifier
# TODO check https://github.com/facebookresearch/moco for correct implementation

class MoCo(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=512, output_dim=128),
        loss_kwargs = dict(memory_bank_size=4096),
        online_linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1),
        online_knn_head_kwargs = dict(num_classes=10, k=20)
    ):
        super().__init__(
            backbone, 
            batch_size_per_device,
            projection_head=MoCoProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            online_linear_head=OnlineLinearClassifier(
                input_dim=backbone.output_dim, 
                **online_linear_head_kwargs
            ),
            online_knn_head=kNNClassifier(
                **online_knn_head_kwargs
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
        self.save_hyperparameters(online_linear_head_kwargs)
        self.save_hyperparameters(online_knn_head_kwargs)
        
    def forward(self, x):
        z = self.backbone(x)
        query = self.projection_head(z)
        return z, query

    def forward_momentum(self, x):
        z = self.backbone_momentum(x)
        key = self.projection_head_momentum(z).detach()
        return z, key    
    
    def training_output(self, batch, batch_index):
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
            x_key, shuffle = batch_shuffle(x_key, distributed=self.is_distributed)
            _, key = self.forward_momentum(x_key)
            key = batch_unshuffle(key, shuffle, distributed=self.is_distributed)
        
        loss = 0.5 * (self.criterion(query, key) + self.criterion(key, query))
        return {
            "loss": loss, 
            "embedding": z_query.detach(),
            "target": y
        }
    
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
                {
                    "name": "moco_weight_decay", 
                    "params": params
                },
                {
                    "name": "moco_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_linear_head.parameters(),
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
    