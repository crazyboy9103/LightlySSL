from lightly.loss.vicreg_loss import VICRegLoss
## The projection head is the same as the Barlow Twins one
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

from .base import BaseModule
from .eval import OnlineClassifier

class VICReg(BaseModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device,
        projection_head_kwargs = dict(hidden_dim=1024, output_dim=256, num_layers=2),
        linear_head_kwargs = dict(num_classes=10, label_smoothing=0.1, k=15),
    ):
        super().__init__(
            backbone, 
            batch_size_per_device,
            projection_head=VICRegProjectionHead(
                input_dim=backbone.output_dim, 
                **projection_head_kwargs
            ),
            linear_head=OnlineClassifier(
                input_dim=backbone.output_dim, 
                **linear_head_kwargs
            )
        )
        
        self.criterion = VICRegLoss(gather_distributed=self.is_distributed)
        
        self.save_hyperparameters(projection_head_kwargs)
        self.save_hyperparameters(linear_head_kwargs)
        
    def forward(self, x):
        z = self.backbone(x).flatten(start_dim=1)
        p = self.projection_head(z)
        return z, p

    def training_step(self, batch, batch_index):
        (x0, x1), y = batch
        z0, p0 = self.forward(x0)
        _, p1 = self.forward(x1)
        loss = self.criterion(p0, p1)
        loss_dict = {"train-ssl-loss": loss}
        
        cls_loss, cls_loss_dict = self.linear_head.training_step((z0.detach(), y), batch_index)
        loss_dict.update(cls_loss_dict)
        self.log_dict(loss_dict, sync_dist=self.is_distributed)         
        return loss + cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        global_batch_size = self.batch_size_per_device * self.trainer.world_size
        base_lr = _get_base_learning_rate(global_batch_size=global_batch_size)
        optimizer = LARS(
            [
                {"name": "vicreg", "params": params},
                {
                    "name": "vicreg_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.linear_head.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Linear learning rate scaling with a base learning rate of 0.2.
            # See https://arxiv.org/pdf/2105.04906.pdf for details.
            lr=base_lr * global_batch_size / 256,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
                end_value=0.01,  # Scale base learning rate from 0.2 to 0.002.
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
    
def _get_base_learning_rate(global_batch_size: int) -> float:
    """Returns the base learning rate for training 100 epochs with a given batch size.

    This follows section C.4 in https://arxiv.org/pdf/2105.04906.pdf.

    """
    if global_batch_size == 128:
        return 0.8
    elif global_batch_size == 256:
        return 0.5
    elif global_batch_size == 512:
        return 0.4
    else:
        return 0.3