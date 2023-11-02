import torchmetrics
import torch
from torch import nn
from torch.nn import functional as F
from lightly.models.utils import deactivate_requires_grad

from .base import BaseModule

class EvalModule(BaseModule):
    def __init__(
        self, 
        backbone, 
        optimizer = torch.optim.SGD, 
        optimizer_kwargs = dict(lr=6e-2), 
        scheduler = None, 
        scheduler_kwargs = None,
        eval_type = "linear",
        num_classes = 10
    ):
        self.eval_type = eval_type
        if eval_type == "linear":
            deactivate_requires_grad(backbone)
            
        super().__init__(
            backbone, 
            optimizer, 
            optimizer_kwargs, 
            scheduler, 
            scheduler_kwargs, 
            linear_head=nn.Linear(backbone.output_dim, num_classes)
        )
        
        metric_kwargs = dict(
            task = "multiclass",
            num_classes = num_classes,
        )
        self.accuracy = torchmetrics.Accuracy(**metric_kwargs)
        self.precision = torchmetrics.Precision(average='weighted', **metric_kwargs)
        self.recall = torchmetrics.Recall(average='weighted', **metric_kwargs)
        self.f1 = torchmetrics.F1Score(average='weighted', **metric_kwargs)
        self.confmat = torchmetrics.ConfusionMatrix(**metric_kwargs) 
        
    def forward(self, x):
        return self.linear(self.backbone(x))
    
    def _step(self, batch, batch_index):
        x, y = batch
        yhat = self.forward(x)
        loss = F.cross_entropy(yhat, y)
        return yhat, loss
    
    def training_step(self, batch, batch_index):
        yhat, loss = self._step(batch, batch_index)
        self.log(f"{self.eval_type}-train-loss", loss, sync_dist=self.is_distributed)
        return loss
    
    def validation_step(self, batch, batch_index):
        yhat, loss = self._step(batch, batch_index)
        self.log(f"{self.eval_type}-valid-loss", loss, sync_dist=self.is_distributed)
        self.accumulate_metrics(yhat, batch[1])
    
    def on_validation_epoch_end(self):
        metrics = self.metrics()
        confmat = metrics.pop("confmat")
        # TODO log & visualize confmat
        # if self.trainer.is_global_zero:
        #     print(confmat)
        
        for k, v in metrics.items():
            self.log(f"valid-{k}", v, sync_dist=self.is_distributed)
            
        self.reset_metrics()
    
    def accumulate_metrics(self, y_hat, y):
        self.accuracy(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.f1(y_hat, y)
        self.confmat(y_hat, y)
    
    def reset_metrics(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confmat.reset()    
        
    def metrics(self):
        return {
            "accuracy": self.accuracy.compute(),
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "f1": self.f1.compute(),
            "confmat": self.confmat.compute()
        }