from typing import List, Dict, Literal
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.accelerators import find_usable_cuda_devices

from dataset import dataset_builder, DataModule
from backbone import backbone_builder
from config import config_builder
from modules import BarlowTwins, BYOL, DINO, MoCo, SimCLR, SwAV, VICReg
from modules import EvalModule

class MetricTracker(pl.Callback):
    r"""
    Automatically logs the maximum/minimum value of a metric. 

    Args:
        metric_config: 
            example = [
                {
                    "name": "train/ssl-loss",
                    "mode": "min",
                    "interval": "step",
                },
                {
                    "name": "valid/online-linear-accuracy",
                    "mode": "max",
                    "interval": "epoch",
                },
            ]

    Example::
        >>> from pytorch_lightning import Trainer
        >>> tracker = MetricTracker(example)
        >>> trainer = Trainer(callbacks=[tracker])
    """
    def __init__(self, metric_config: List[Dict[str, str]]):
        super().__init__()
        for config in metric_config:
            assert "name" in config
            assert "mode" in config and config["mode"] in ["max", "min"]
            assert "interval" in config and config["interval"] in ["epoch", "step"]
            config.setdefault("value", -1e8 if config["mode"] == "max" else 1e8)
            
        self.epoch_config = [config for config in metric_config if config["interval"] == "epoch"]
        self.step_config = [config for config in metric_config if config["interval"] == "step"]
    
    def _log_metrics(self, trainer, interval: Literal["step", "epoch"]="step"):
        metrics_to_log = {}
        
        config = self.step_config if interval == "step" else self.epoch_config
        metrics = trainer.callback_metrics
        for cfg in config:
            name = cfg["name"]
            mode = cfg["mode"]
            
            if name in metrics:
                cfg["value"] = max(cfg["value"], metrics[name]) if mode == "max" else min(cfg["value"], metrics[name])
                metrics_to_log[f"{mode}_{name}"] = cfg["value"]
                
        if metrics_to_log and trainer.logger:
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step if interval == "step" else trainer.current_epoch)
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        self._log_metrics(trainer, interval="step")
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_metrics(trainer, interval="epoch")
        
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx = 0):
        self._log_metrics(trainer, interval="step")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_metrics(trainer, interval="epoch")

def trainer_builder(
    devices,
    checkpoint_path, 
    logger, 
    metric_name,
    metric_mode, 
    epochs
):
    # os.makedirs(checkpoint_path, exist_ok=True)
    
    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=epochs,
        precision="16-mixed",
        benchmark=True,
        callbacks=[
            MetricTracker([{"name": metric_name, "mode": metric_mode, "interval": "epoch"}]),
            ModelCheckpoint(dirpath=checkpoint_path, save_top_k=1, monitor=metric_name, mode=metric_mode, auto_insert_metric_name=False),
            ModelSummary(max_depth=-1),
            LearningRateMonitor(logging_interval="step")
        ],
        fast_dev_run=False,
        sync_batchnorm=len(devices) > 1,
        devices=devices,
        log_every_n_steps=1,
        strategy="ddp" if len(devices) > 1 else "auto", 
        num_sanity_val_steps=0, # must be 0 
        use_distributed_sampler = True,
    )
    return trainer

def main(args):
    model_config = config_builder(args)
    devices = find_usable_cuda_devices(args.num_gpus)
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    # torch.set_float32_matmul_precision('high')
    
    backbone = backbone_builder(
        args.backbone, 
        args.backbone_checkpoint
    )
    
    train_data, test_data, train_transform, test_transform = dataset_builder(
        args.ssl, 
        args.dataset, 
        args.data_root, 
    )

    experiment_name = f'{args.ssl}_{args.backbone}_{args.dataset}_noreset'
    logger = WandbLogger(
        project="ssl-lightly",
        name=experiment_name,
        log_model=False,
        save_dir="."
    ) if args.wandb else TensorBoardLogger(
        save_dir="./tb_logs",
        name=experiment_name,
        default_hp_metric=False
    )
    
    def pretrain():
        models = {
            "barlowtwins": BarlowTwins,
            "byol": BYOL,
            "dino": DINO,
            "moco": MoCo,
            "simclr": SimCLR,
            "swav": SwAV,
            "vicreg": VICReg,
        }
        
        model = models[args.ssl](backbone, args.batch_size, **model_config)
        
        
        # model = torch.compile(model)
        pretrainer = trainer_builder(
            devices,
            f'./checkpoints/ssl/{args.ssl}/{args.backbone}/{args.dataset}/ssl_model', 
            logger,
            "valid/online-linear-accuracy", # "train/ssl-loss", "train/online-linear-loss", "valid/online-linear-loss", "train/online-linear-accuracy", "valid/online-linear-accuracy"
            "max",
            args.pretrain_epochs
        )
    
        train_data.transform = train_transform
        test_data.transform = test_transform

        data_loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        
        datamodule = DataModule(train_data, test_data, data_loader_kwargs)
        
        pretrainer.fit(
            model, 
            datamodule=datamodule
        )
        
    def evaluate():
        evaluator = trainer_builder(
            devices,
            f'./checkpoints/sl/{args.sl}/{args.backbone}/{args.dataset}/sl_model', 
            logger,
            "valid/linear-accuracy", # "train/linear-loss", "valid/linear-loss", "train/linear-accuracy", "valid/linear-accuracy"
            "max",
            args.eval_epochs
        )
    
        train_data.transform = test_transform
        test_data.transform = test_transform

        data_loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        
        datamodule = DataModule(train_data, test_data, data_loader_kwargs)
        
        model = EvalModule(
            backbone, 
            args.batch_size,
            model_config["online_linear_head_kwargs"]["num_classes"],
            args.sl,
            args.label_smoothing,
            args.k
        )
        evaluator.fit(
            model, 
            datamodule=datamodule
        )
        
    if "train" in args.experiment:
        pretrain()
    
    if "eval" in args.experiment:
        evaluate()

if __name__ == "__main__":
    import argparse

    from backbone import AVAILABLE_BACKBONES

    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_root", type=str, default="./data")
    
    # model args
    parser.add_argument("--backbone", type=str, default="resnet18", choices=AVAILABLE_BACKBONES)
    parser.add_argument("--backbone_checkpoint", type=str, default="")
    # training args
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--experiment", type=str, default="train+eval")
    # pretrain args
    parser.add_argument("--pretrain_epochs", type=int, default=400)
    parser.add_argument("--ssl", type=str, default="byol", choices=["barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"])
    # eval args
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--sl", type=str, default="linear", choices=["linear", "finetune"])
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors for kNN")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    # misc
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--wandb", action="store_true", default=True)

    args = parser.parse_args()

    # each device sees a batch size divided by the number of devices
    # as lightning sends same number of samples to each device
    args.batch_size = args.batch_size // args.num_gpus
    main(args)