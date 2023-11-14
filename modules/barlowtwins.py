from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

from .base import BaseModule
from .eval import OnlineLinearClassifier

class BarlowTwins(BaseModule):
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
            projection_head=BarlowTwinsProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            online_linear_head=OnlineLinearClassifier(
                input_dim=backbone.output_dim, 
                **online_linear_head_kwargs
            )
        )
        
        self.criterion = BarlowTwinsLoss(gather_distributed=self.is_distributed)
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(online_linear_head_kwargs)
        
    def forward(self, x):
        z = self.backbone(x)
        p = self.projection_head(z)
        return z, p

    def training_output(self, batch, batch_index):
        (x0, x1), y = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = self.criterion(p0, p1)
        return {
            "loss": loss, 
            "embedding": z0.detach(),
            "target": y
        }
    
    def configure_optimizers(self):
        lr_factor = self.batch_size_per_device * self.trainer.world_size / 256

        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = LARS(
            [
                {
                    "name": "barlowtwins_weight_decay", 
                    "params": params
                },
                {
                    "name": "barlowtwins_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                    "lr": 0.0048 * lr_factor,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_linear_head.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.2 * lr_factor,
            momentum=0.9,
            weight_decay=1.5e-6,
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