from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from lightly.loss import SwaVLoss
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

from .base import BaseModule
from .eval import OnlineLinearClassifier, OnlinekNNClassifier

class SwAV(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=512, output_dim=128),
        prototype_kwargs = dict(n_prototypes=512),
        online_linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1),
        online_knn_head_kwargs = dict(num_classes=10, k=20),
        optimizer_kwargs = dict(base_lr=0.6)
    ):
        super().__init__(
            backbone, 
            batch_size_per_device,
            projection_head=SwaVProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            prototypes=SwaVPrototypes(
                input_dim=projection_head_kwargs["output_dim"], 
                **prototype_kwargs
            ),
            online_linear_head=OnlineLinearClassifier(
                input_dim=backbone.output_dim, 
                **online_linear_head_kwargs
            ),
            online_knn_head=OnlinekNNClassifier(
                **online_knn_head_kwargs
            )
        )
        
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=self.is_distributed)
        
        # Use a queue for small batch sizes (<= 256).
        self.start_queue_at_epoch = 15
        self.n_batches_in_queue = 15
        self.queues = nn.ModuleList(
            [
                MemoryBankModule(
                    size=self.n_batches_in_queue * self.batch_size_per_device
                )
                for _ in range(2)
            ]
        )
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(prototype_kwargs)
        self.save_hyperparameters(online_linear_head_kwargs)
        self.save_hyperparameters(online_knn_head_kwargs)
        self.save_hyperparameters(optimizer_kwargs)
        
    def forward(self, x):
        z = self.backbone(x)
        p = self.projection_head(z)
        proj = F.normalize(p, dim=1, p=2)
        proto = self.prototypes(proj, step=self.current_epoch)
        return z, proj, proto

    def training_output(self, batch, batch_index):
        self.prototypes.normalize()
        views, y = batch
        outputs = [self.forward(view) for view in views]
        features, projections, prototypes = zip(*outputs)
        
        # Get the queue projections and logits.
        queue_prototypes = None
        with torch.no_grad():
            if self.current_epoch >= self.start_queue_at_epoch:
                # Start filling the queue.
                queue_crop_projections = self._update_queue(
                    projections=projections[:2],
                    queues=self.queues,
                )
                if batch_index > self.n_batches_in_queue:
                    # The queue is filled, so we can start using it.
                    queue_prototypes = [
                        self.prototypes(projections, step=self.current_epoch)
                        for projections in queue_crop_projections
                    ]
        
        high_resolution = prototypes[:2]
        low_resolution = prototypes[2:]
        loss = self.criterion(
            high_resolution_outputs=high_resolution, 
            low_resolution_outputs=low_resolution, 
            queue_outputs=queue_prototypes
        )
        return {
            "loss": loss, 
            "embedding": features[0].detach(),
            "target": y
        }
    
    
    @torch.no_grad()
    def _update_queue(
        self,
        projections: List[torch.Tensor],
        queues: nn.ModuleList,
    ):
        """Adds the high resolution projections to the queues and returns the queues."""

        if len(projections) != len(queues):
            raise ValueError(
                f"The number of queues ({len(queues)}) should be equal to the number of high "
                f"resolution inputs ({len(projections)})."
            )

        # Get the queue projections
        queue_projections = []
        for i in range(len(queues)):
            _, queue_proj = queues[i](projections[i], update=True)
            # Queue projections are in (num_ftrs X queue_length) shape, while the high res
            # projections are in (batch_size_per_device X num_ftrs). Swap the axes for interoperability.
            queue_proj = torch.permute(queue_proj, (1, 0))
            queue_projections.append(queue_proj)

        return queue_projections

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head, self.prototypes]
        )
        optimizer = LARS(
            [
                {
                    "name": "swav_weight_decay", 
                    "params": params
                },
                {
                    "name": "swav_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_linear_head.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Smaller learning rate for smaller batches: lr=0.6 for batch_size=256
            # scaled linearly by batch size to lr=4.8 for batch_size=2048.
            # See Appendix A.1. and A.6. in SwAV paper https://arxiv.org/pdf/2006.09882.pdf
            lr=self.hparams.base_lr * (self.batch_size_per_device * self.trainer.world_size) / 256,
            momentum=0.9,
            weight_decay=1e-6,
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
                end_value=0.0006
                * (self.batch_size_per_device * self.trainer.world_size)
                / 256,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]