import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl

class OnlineClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 2048,
        num_classes: int = 1000,
        label_smoothing: float = 0.0,
        k: int = 15,
    ) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)
        self.label_smoothing = label_smoothing
        self.k = k
        
        metric_kwargs = dict(
            task = "multiclass",
            num_classes = num_classes,
        )

        # for linear only (finetune excluded in on-the-fly eval for complexity)
        self.accuracy = torchmetrics.Accuracy(**metric_kwargs)
        self.f1 = torchmetrics.F1Score(average='weighted', **metric_kwargs)
        
        self.knn_accuracy = torchmetrics.Accuracy(**metric_kwargs)
        self.knn_f1 = torchmetrics.F1Score(average='weighted', **metric_kwargs)
        
        self.z_train = []
        self.y_train = []
    
    def on_train_epoch_end(self):
        return self._epoch_end("train")
        
    def on_validation_epoch_end(self):
        # After validation, we would want to reset the parameters of the head (linear probe)
        self.head.reset_parameters()
        return self._epoch_end("valid")
        
    def _epoch_end(self, phase):
        metrics = self.metrics(phase)
        self.reset_metrics(phase)
        return metrics 
            
    def accumulate_knn(self, z_hat, y):
        # at train time, we accumulate the embeddings and labels
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
            
            self.knn_accuracy(knn_yhat, y)
            self.knn_f1(knn_yhat, y)
            
    def accumulate_linear(self, y_hat, y):
        self.accuracy(y_hat, y)
        self.f1(y_hat, y)
        
    def forward(self, z_hat, y):
        z_hat = z_hat.detach().flatten(start_dim=1)
        y_hat = self.head(z_hat)
        loss = F.cross_entropy(y_hat, y, label_smoothing=self.label_smoothing if self.training else 0.0)
        return loss, y_hat, z_hat
    
    def training_step(self, batch, batch_idx):
        z_hat, y = batch
        loss, y_hat, z_hat = self.forward(z_hat, y)
        self.accumulate_linear(y_hat, y)
        self.accumulate_knn(z_hat, y)
        return loss, {"onlinelinear-train-loss": loss}

    def validation_step(self, batch, batch_idx):
        z_hat, y = batch
        loss, y_hat, z_hat = self.forward(z_hat, y)
        self.accumulate_linear(y_hat, y)
        self.accumulate_knn(z_hat, y)
        return loss, {"onlinelinear-valid-loss": loss}
    
    def reset_metrics(self, phase):
        if phase == "train":
            self.accuracy.reset()
            self.f1.reset()
        
        elif phase == "valid":
            self.accuracy.reset()
            self.f1.reset()
            self.knn_accuracy.reset()
            self.knn_f1.reset()
            self.z_train = []
            self.y_train = []
        
    def metrics(self, phase):
        # for train, we only compute linear/finetune metrics
        if phase == "train":
            return {
                "accuracy": self.accuracy.compute(),
                "f1": self.f1.compute()
            }
        
        # for validation, we compute knn metrics too 
        elif phase == "valid":
            return {
                "accuracy": self.accuracy.compute(),
                "f1": self.f1.compute(),
                "knn_accuracy": self.knn_accuracy.compute(),
                "knn_f1": self.knn_f1.compute()
            }

# TODO: may not be needed
# class EvalModule(BaseModule):
#     def __init__(
#         self, 
#         backbone, 
#         optimizer = torch.optim.SGD, 
#         optimizer_kwargs = dict(lr=6e-2), 
#         scheduler = None, 
#         scheduler_kwargs = None,
#         eval_type = "linear",
#         num_classes = 10,
#         label_smoothing: float = 0.0,        
#         k: int = 15,
#     ):
#         if eval_type == "linear":
#             deactivate_requires_grad(backbone)
            
#         super().__init__(
#             backbone, 
#             optimizer, 
#             optimizer_kwargs, 
#             scheduler, 
#             scheduler_kwargs, 
#             linear_head=OnlineClassifier(
#                 backbone.output_dim, 
#                 num_classes,
#                 label_smoothing,
#                 k,
#                 eval_type
#             )
#         )

#     def forward(self, x):
#         zhat = self.backbone(x)
#         return zhat
    
#     def training_step(self, batch, batch_index):
#         zhat = self.forward(batch[0])
#         loss, loss_dict = self.linear.training_step(zhat, batch[1], batch_index)
#         self.log_dict(loss_dict, sync_dist=self.is_distributed)
#         return loss

#     def on_train_epoch_end(self):
#         metrics = self.linear.on_train_epoch_end()
#         self.log_dict(metrics, sync_dist=self.is_distributed)
    
#     def on_validation_epoch_end(self):
#         metrics = self.linear.on_validation_epoch_end()
#         self.log_dict(metrics, sync_dist=self.is_distributed)
        
#     def validation_step(self, batch, batch_index):
#         zhat = self.forward(batch[0])
#         _, loss_dict = self.linear.validation_step(zhat, batch[1], batch_index)
#         self.log_dict(loss_dict, sync_dist=self.is_distributed)
    
if __name__ == "__main__":
    classifier = OnlineClassifier(
        input_dim=2048,
        num_classes=1000,
        label_smoothing=0.0,
        k=15,
    )
    for param in classifier.parameters():
        print(param.shape, param.requires_grad)