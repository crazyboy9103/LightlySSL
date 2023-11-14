from typing import Literal

from lightly.models.utils import (
    deactivate_requires_grad, 
    activate_requires_grad,
)
from lightly.utils.scheduler import CosineWarmupScheduler

import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import pytorch_lightning as pl

class kNNClassifier(pl.LightningModule):
    def __init__(
        self, 
        num_classes: int = 1000,    
        k: int = 15
    ):
        super().__init__()
        self.num_classes = num_classes
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
        
        self.z_train_gathered = None
        self.y_train_gathered = None

    def knn_predict(self, feature, feature_bank, feature_labels, criterion: Literal["cosine", "euclidean"] = "cosine"):
        """Helper method to run kNN predictions on features based on a feature bank

        Args:
            feature: [B, D] consisting of N D-dimensional features
            feature_bank: [N, D] Tensor of a database of features used for kNN
            feature_labels: Labels for the features in our feature_bank
        """
        if criterion == "cosine":
            # # normalize embeddings for 
            # feature, feature_bank = F.normalize(feature), F.normalize(feature_bank)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            dists = pairwise_cosine_similarity(feature, feature_bank) 
        
        elif criterion == "euclidean": 
            dists = torch.cdist(feature, feature_bank, p=2)
            
        k = min(self.k, len(feature_bank))
        knn_dists, knn_idxs = torch.topk(dists, k, dim=1, largest=False if criterion == "euclidean" else True)
        knn_yhat = feature_labels[knn_idxs].mode(dim=1).values
        return knn_yhat
    
    def accumulate(self, z_hat, y):
        if self.training:
            self.z_train.append(z_hat.detach())
            self.y_train.append(y.detach())
        
        # at validation time, we add knn predictions to the torchmetrics 
        else:
            if self.z_train_gathered == None and self.y_train_gathered == None:
                self.z_train_gathered = self.all_gather(torch.cat(self.z_train))
                self.y_train_gathered = self.all_gather(torch.cat(self.y_train))
                self.z_train_gathered = self.z_train_gathered.view(-1, self.z_train_gathered.shape[-1])
                self.y_train_gathered = self.y_train_gathered.view(-1)

            knn_yhat = self.knn_predict(z_hat.detach(), self.z_train_gathered, self.y_train_gathered)

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
        
        self.z_train_gathered = None
        self.y_train_gathered = None
        
    def metrics(self):
        # only valid for knn
        return {
            "valid/knn-accuracy": self.accuracy.compute(),
            "valid/knn-f1": self.f1.compute()
        }

# Re-written classifier, since we want to have separate metrics for train and valid
# This only records epoch-wise metrics, to avoid class imbalance issues with mini-batch metrics
# To do this, it accumulates preds/targets at every training/validation step, and computes metrics at every on_train/valid_epoch_end
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
        # we keep separate metrics for train and valid, since the order of the hooks 
        # is: on_train_epoch_start()->training_step()
        #   ->on_validation_epoch_start()->validation_step()->on_validation_epoch_end()
        #   ->on_train_epoch_end()
        self.train_accuracy = torchmetrics.Accuracy(**metric_kwargs)
        self.train_f1 = torchmetrics.F1Score(average='macro', **metric_kwargs)

        self.valid_accuracy = torchmetrics.Accuracy(**metric_kwargs)
        self.valid_f1 = torchmetrics.F1Score(average='macro', **metric_kwargs)
        
    @property
    def phase(self):
        return "train" if self.training else "valid"
    
    @property
    def name(self):
        return self.eval_type

    def on_train_epoch_end(self):
        return self._epoch_end()

    def on_validation_epoch_end(self):
        return self._epoch_end()
        
    def _epoch_end(self):
        metrics = self.metrics()
        self.reset_metrics()
        return metrics 
            
    def accumulate(self, y_hat, y):
        accuracy, f1 = self._metrics()
        
        accuracy(y_hat, y)
        f1(y_hat, y)
        
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    def _step(self, batch, batch_index):
        z_hat, y = batch
        
        if self.eval_type == "linear":
            z_hat = z_hat.detach()
            
        y_hat = self.head(z_hat)
        loss = F.cross_entropy(y_hat, y, label_smoothing=self.label_smoothing if self.training else 0.0)

        self.accumulate(y_hat, y)
        return loss, {f"{self.phase}/{self.name}-loss": loss}
    
    def reset_metrics(self):
        accuracy, f1 = self._metrics()
        accuracy.reset()
        f1.reset()
    
    def _metrics(self):
        accuracy = getattr(self, f"{self.phase}_accuracy")
        f1 = getattr(self, f"{self.phase}_f1")
        return accuracy, f1
    
    def metrics(self):
        accuracy, f1 = self._metrics()
        return {
            f"{self.phase}/{self.name}-accuracy": accuracy.compute(),
            f"{self.phase}/{self.name}-f1": f1.compute()
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
        return super().on_validation_epoch_end()
    
    @property
    def name(self):
        return "online-linear"
        
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
        self.is_distributed = torch.cuda.device_count() > 1
        
        # TODO: may not need this
        if eval_type == "linear":
            deactivate_requires_grad(self.backbone)
            self.backbone.eval()

        elif eval_type == "finetune":
            activate_requires_grad(self.backbone)
            self.backbone.train()
        
        self.eval_type = eval_type
        
    @property
    def should_knn_eval(self):
        # for linear probe, we only evaluate once, at the start of the epoch, 
        # as the backbone is not updated during training
        if self.eval_type == "linear":
            return self.current_epoch == 0
        
        return True
    
    def training_step(self, batch, batch_index):
        x, y = batch
        zhat = self.backbone(x)
        loss, loss_dict = self.linear_head.training_step([zhat, y], batch_index)
        if self.should_knn_eval:
            self.knn_head.training_step([zhat, y], batch_index)
            
        self.log_dict(loss_dict, sync_dist=self.is_distributed)
        return loss

    def on_train_epoch_end(self):
        linear_metrics = self.linear_head.on_train_epoch_end()
        self.log_dict(linear_metrics, sync_dist=self.is_distributed)
    
    def on_validation_epoch_end(self):
        linear_metrics = self.linear_head.on_validation_epoch_end()
        if self.should_knn_eval:
            knn_metrics = self.knn_head.on_validation_epoch_end()
            linear_metrics.update(knn_metrics)
            
        self.log_dict(linear_metrics, sync_dist=self.is_distributed)
        
    def validation_step(self, batch, batch_index):
        x, y = batch
        zhat = self.backbone(x)
        _, loss_dict = self.linear_head.validation_step([zhat, y], batch_index)
        if self.should_knn_eval:
            self.knn_head.validation_step([zhat, y], batch_index)
            
        self.log_dict(loss_dict, sync_dist=self.is_distributed)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()),
            # Settings follow original code for 100 epochs which are slightly different
            # from the paper, see:
            # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
            lr=0.1 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

if __name__ == "__main__":
    classifier = LinearClassifier(
        input_dim=2048,
        num_classes=1000,
        label_smoothing=0.0,
    )
    print(classifier.__class__.__name__)