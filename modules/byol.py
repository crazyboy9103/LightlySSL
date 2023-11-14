import copy 

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import (
    deactivate_requires_grad, 
    update_momentum,
    get_weight_decay_parameters
)
from lightly.utils.lars import LARS
from lightly.utils.scheduler import cosine_schedule
from lightly.utils.scheduler import CosineWarmupScheduler

from .base import BaseModule
from .eval import OnlineLinearClassifier, kNNClassifier

class BYOL(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=1024, output_dim=256),
        prediction_head_kwargs = dict(hidden_dim=1024, output_dim=256),
        online_linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1),
        online_knn_head_kwargs = dict(num_classes=10, k=20)
    ):
        super().__init__(
            backbone, 
            batch_size_per_device,
            projection_head=BYOLProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            prediction_head=BYOLPredictionHead(
                input_dim=projection_head_kwargs["output_dim"],  
                **prediction_head_kwargs
            ),
            online_linear_head=OnlineLinearClassifier(
                input_dim=backbone.output_dim, 
                **online_linear_head_kwargs
            ),
            online_knn_head=kNNClassifier(
                **online_knn_head_kwargs
            )
        )
        
        self.backbone_momentum = copy.deepcopy(backbone)
        self.projection_head_momentum = BYOLProjectionHead(
            input_dim=backbone.output_dim, 
            **projection_head_kwargs
        )
        
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        
        self.criterion = NegativeCosineSimilarity()
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(prediction_head_kwargs)
        self.save_hyperparameters(online_linear_head_kwargs)
        self.save_hyperparameters(online_knn_head_kwargs)
        
    def forward(self, x):
        # here we use different notation from the paper to maintain consistency
        # z: embedding, y: projection, p: prediction
        y = self.backbone(x)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return y, p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    
    def training_output(self, batch, batch_index):
        momentum = cosine_schedule(
            self.trainer.global_step, 
            self.trainer.estimated_stepping_batches, 
            start_value=0.99, 
            end_value=1
        )
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        (x0, x1), y = batch
        y0, p0 = self.forward(x0)
        y1, p1 = self.forward(x1)
        z0 = self.forward_momentum(x0)
        z1 = self.forward_momentum(x1)
        loss = 2 * (self.criterion(p0, z1) + self.criterion(p1, z0)) # truncate constant 2 from loss (2-2cos(p, z))
        return {
            "loss": loss, 
            "embedding": y0.detach(),
            "target": y
        }

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [
                self.backbone,
                self.projection_head,
                self.prediction_head,
            ]
        )
        optimizer = LARS(
            [
                {
                    "name": "byol_weight_decay", 
                    "params": params
                },
                {
                    "name": "byol_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_linear_head.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Settings follow original code for 100 epochs which are slightly different
            # from the paper, see:
            # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
            lr=0.45 * self.batch_size_per_device * self.trainer.world_size / 256,
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
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
