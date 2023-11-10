from torch.nn import functional as F
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

from .base import BaseModule
from .eval import OnlineLinearClassifier
# TODO add queue
class SwAV(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=512, output_dim=128),
        prototype_kwargs = dict(n_prototypes=512),
        online_linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1),
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
            )
        )
        
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=self.is_distributed)
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(prototype_kwargs)
        self.save_hyperparameters(online_linear_head_kwargs)
        
    def forward(self, x):
        z = self.backbone(x).flatten(start_dim=1)
        p = self.projection_head(z)
        p = F.normalize(p, dim=1, p=2)
        p = self.prototypes(p)
        return z, p

    def training_output(self, batch, batch_index):
        self.prototypes.normalize()
        views, y = batch
        outputs = [self.forward(view) for view in views]
        features, prototypes = zip(*outputs)
        high_resolution = prototypes[:2]
        low_resolution = prototypes[2:]
        loss = self.criterion(high_resolution, low_resolution)
        return {
            "loss": loss, 
            "embedding": features[0].detach(),
            "target": y
        }
    
    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head, self.prototypes]
        )
        optimizer = LARS(
            [
                {"name": "swav", "params": params},
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
            lr=0.6 * (self.batch_size_per_device * self.trainer.world_size) / 256,
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