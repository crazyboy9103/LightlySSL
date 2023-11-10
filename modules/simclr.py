from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

from .base import BaseModule
from .eval import OnlineLinearClassifier

class SimCLR(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=2048, output_dim=2048),
        online_linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1),
    ):
        super().__init__(
            backbone, 
            batch_size_per_device,
            projection_head=SimCLRProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            online_linear_head=OnlineLinearClassifier(
                input_dim=backbone.output_dim, 
                **online_linear_head_kwargs
            )
        )
        self.criterion = NTXentLoss(gather_distributed=self.is_distributed)
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(online_linear_head_kwargs)
        
    def forward(self, x):
        z = self.backbone(x).flatten(start_dim=1)
        p = self.projection_head(z)
        return z, p

    def training_output(self, batch, batch_index):
        (x0, x1), y = batch
        z0, p0 = self.forward(x0)
        _, p1 = self.forward(x1)
        loss = self.criterion(p0, p1)
        return {
            "loss": loss, 
            "embedding": z0.detach(),
            "target": y
        }
    
    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = LARS(
            [
                {"name": "simclr", "params": params},
                {
                    "name": "simclr_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_linear_head.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Square root learning rate scaling improves performance for small
            # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
            # linear scaling can be used for larger batches and longer training:
            #   lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256
            # See Appendix B.1. in the SimCLR paper https://arxiv.org/abs/2002.05709
            lr=0.075 * (self.batch_size_per_device * self.trainer.world_size) ** 0.5,
            momentum=0.9,
            # Note: Paper uses weight decay of 1e-6 but reference code 1e-4. See:
            # https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/README.md?plain=1#L103
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
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]