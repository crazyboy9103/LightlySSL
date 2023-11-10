from typing import Literal

from lightly.models.utils import deactivate_requires_grad

import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl

class kNNClassifier(pl.LightningModule):
    def __init__(
        self, 
        num_classes: int = 1000,    
        k: int = 15
    ):
        super().__init__()
        self.k = k
        
        metric_kwargs = dict(
            task = "multiclass",
            num_classes = num_classes,
        )
        
        self.accuracy = torchmetrics.Accuracy(**metric_kwargs)
        self.f1 = torchmetrics.F1Score(average='macro', **metric_kwargs)
        
        # we accumulate the embeddings and labels for kNN predictions
        self.z_train = []
        self.y_train = []
    
    def accumulate(self, z_hat, y):
        if self.training:
            self.z_train.append(z_hat.detach())
            self.y_train.append(y.detach())
        
        # at validation time, we add knn predictions to the metrics 
        else:
            neighbors = torch.cat(self.z_train)
            labels = torch.cat(self.y_train)
            
            dists = torch.cdist(z_hat.detach(), neighbors, p=2)
            
            k = min(self.k, len(neighbors))
            knn_dists, knn_idxs = torch.topk(dists, k, dim=1, largest=False)
            knn_yhat = labels[knn_idxs].mode(dim=1).values
            
            self.accuracy(knn_yhat, y)
            self.f1(knn_yhat, y)
    
    def training_step(self, batch, batch_idx):
        self.accumulate(*batch)

    def validation_step(self, batch, batch_idx):
        self.accumulate(*batch)
    
    def on_validation_epoch_end(self):
        metrics = self.metrics()
        self.reset_metrics()
        return metrics 
    
    def reset_metrics(self):
        self.accuracy.reset()
        self.f1.reset()
        
        self.z_train = []
        self.y_train = []
        
    def metrics(self):
        # only valid for knn
        return {
            "valid/knn-accuracy": self.accuracy.compute(),
            "valid/knn-f1": self.f1.compute()
        }

class LinearClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 2048,
        num_classes: int = 1000,
        label_smoothing: float = 0.0,
        eval_type: Literal["linear", "finetune"] = "linear",
    ):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)
        self.label_smoothing = label_smoothing
        self.eval_type = eval_type
        
        metric_kwargs = dict(
            task = "multiclass",
            num_classes = num_classes,
        )

        # for linear only (kNN, finetune excluded from on-the-fly eval)
        self.accuracy = torchmetrics.Accuracy(**metric_kwargs)
        self.f1 = torchmetrics.F1Score(average='macro', **metric_kwargs)
        
    def on_train_epoch_end(self):
        metrics = self._epoch_end()
        return {
            f"train/{k}": v for k, v in metrics.items()
        }
        
    def on_validation_epoch_end(self):
        metrics = self._epoch_end()
        return {
            f"valid/{k}": v for k, v in metrics.items()
        }
        
    def _epoch_end(self):
        metrics = self.metrics()
        self.reset_metrics()
        return metrics 
            
    def accumulate(self, y_hat, y):
        self.accuracy(y_hat, y)
        self.f1(y_hat, y)
        
    def forward(self, z_hat, y):
        if self.eval_type == "linear":
            z_hat = z_hat.detach()
            
        y_hat = self.head(z_hat.flatten(start_dim=1))
        loss = F.cross_entropy(y_hat, y, label_smoothing=self.label_smoothing if self.training else 0.0)
        return loss, y_hat
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        return loss, {f"train/{self.eval_type}-loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        return loss, {f"valid/{self.eval_type}-loss": loss}
    
    def _step(self, batch, batch_index):
        z_hat, y = batch
        loss, y_hat = self.forward(z_hat, y)
        self.accumulate(y_hat, y)
        return loss
    
    def reset_metrics(self):
        self.accuracy.reset()
        self.f1.reset()
        
    def metrics(self):
        return {
            f"{self.eval_type}-accuracy": self.accuracy.compute(),
            f"{self.eval_type}-f1": self.f1.compute()
        }

class OnlineLinearClassifier(LinearClassifier):
    def __init__(
        self,
        input_dim: int = 2048,
        num_classes: int = 1000,
        label_smoothing: float = 0.0,
    ):
        super().__init__(
            input_dim = input_dim, 
            num_classes = num_classes, 
            label_smoothing = label_smoothing, 
            eval_type = "linear"
        )
        
    def on_validation_epoch_end(self):
        # After validation, we would want to reset the parameters of the head (linear probe)
        self.head.reset_parameters()
        return self._epoch_end()
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = super().training_step(batch, batch_idx)
        return loss, {"train/online-linear-loss": loss_dict["train/linear-loss"]}

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = super().validation_step(batch, batch_idx)
        return loss, {"valid/online-linear-loss": loss_dict["valid/linear-loss"]}
    
    def metrics(self):
        return {
            "online-linear-accuracy": self.accuracy.compute(),
            "online-linear-f1": self.f1.compute()
        }
        
class EvalModule(pl.LightningModule):
    def __init__(
        self, 
        backbone, 
        batch_size_per_device: int, 
        num_classes: int = 1000,
        eval_type: Literal["linear", "finetune"] = "linear",
        label_smoothing: float = 0.0,
        k: int = 15,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.batch_size_per_device = batch_size_per_device
        self.linear_head = LinearClassifier(
            input_dim = backbone.output_dim,
            num_classes = num_classes,
            label_smoothing = label_smoothing,
            eval_type = eval_type,
        )
        self.knn_head = kNNClassifier(
            num_classes = num_classes,
            k = k,
        )
        
        # TODO: may not need this
        if eval_type == "linear":
            deactivate_requires_grad(self.backbone)
            self.backbone.eval()

    def training_step(self, batch, batch_index):
        x, y = batch
        zhat = self.backbone(x)
        loss, loss_dict = self.linear_head.training_step(zhat, y, batch_index)
        self.knn_head.training_step(zhat, y, batch_index)
        self.log_dict(loss_dict, sync_dist=self.is_distributed)
        return loss

    def on_train_epoch_end(self):
        linear_metrics = self.linear_head.on_train_epoch_end()
        knn_metrics = self.knn_head.on_train_epoch_end()
        self.log_dict({**linear_metrics, **knn_metrics}, sync_dist=self.is_distributed)
    
    def on_validation_epoch_end(self):
        linear_metrics = self.linear_head.on_validation_epoch_end()
        knn_metrics = self.knn_head.on_validation_epoch_end()
        self.log_dict({**linear_metrics, **knn_metrics}, sync_dist=self.is_distributed)
        
    def validation_step(self, batch, batch_index):
        x, y = batch
        zhat = self.backbone(x)
        _, loss_dict = self.linear_head.validation_step(zhat, y, batch_index)
        self.knn_head.validation_step(zhat, y, batch_index)
        self.log_dict(loss_dict, sync_dist=self.is_distributed)
    
if __name__ == "__main__":
    classifier = LinearClassifier(
        input_dim=2048,
        num_classes=1000,
        label_smoothing=0.0,
    )
    print(classifier.__class__.__name__)