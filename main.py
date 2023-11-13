import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.accelerators import find_usable_cuda_devices

from dataset import dataset_builder, DataModule
from backbone import backbone_builder
from config import config_builder

from modules import BarlowTwins, BYOL, DINO, MoCo, SimCLR, SwAV, VICReg
from modules import EvalModule

def trainer_builder(
    devices,
    checkpoint_path, 
    logger, 
    metric_name,
    metric_mode, 
    epochs
):
    os.makedirs(checkpoint_path, exist_ok=True)
    
    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=epochs,
        precision="32",
        benchmark=True,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor=metric_name, mode=metric_mode),
            ModelSummary(max_depth=-1),
            # EarlyStopping(monitor=metric_name, patience=10, mode=metric_mode, verbose=True)
        ],
        fast_dev_run = False,
        sync_batchnorm=True,
        devices=devices,
        log_every_n_steps=1,
        strategy="auto", # "ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        use_distributed_sampler = True,
    )
    return trainer

def main(args):
    model_config = config_builder(args)
    devices = find_usable_cuda_devices(args.num_gpus)
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    
    backbone = backbone_builder(
        args.backbone, 
        args.backbone_checkpoint
    )
    
    train_data, test_data, train_transform, test_transform = dataset_builder(
        args.ssl, 
        args.dataset, 
        args.data_root, 
    )

    experiment_name = f'{args.ssl}_{args.backbone}_{args.dataset}'
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
            f'./checkpoints/ssl/{args.ssl}/{args.backbone}/{args.dataset}', 
            logger,
            "train/ssl-loss", # "train/ssl-loss", "train/online-linear-loss", "valid/online-linear-loss", "train/online-linear-accuracy", "valid/online-linear-accuracy"
            "min",
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
            f'./checkpoints/sl/{args.sl}/{args.backbone}/{args.dataset}', 
            logger,
            "valid/linear-loss", # "train/linear-loss", "valid/linear-loss", "train/linear-accuracy", "valid/linear-accuracy"
            "min",
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
    parser.add_argument("--backbone", type=str, default="resnet50", choices=AVAILABLE_BACKBONES)
    parser.add_argument("--backbone_checkpoint", type=str, default="./checkpoints/ssl/barlowtwins/resnet50/cifar10/epoch=325-step=31948.ckpt")
    # training args
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--experiment", type=str, default="eval")
    # pretrain args
    parser.add_argument("--pretrain_epochs", type=int, default=400)
    parser.add_argument("--ssl", type=str, default="barlowtwins", choices=["barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"])
    # eval args
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--sl", type=str, default="linear", choices=["linear", "finetune"])
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors for kNN")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    # misc
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--wandb", action="store_true", default=True)

    args = parser.parse_args()

    main(args)