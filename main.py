import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from dataset import dataset_builder
from backbone import backbone_builder
from config import config_builder

from modules import BarlowTwins, BYOL, DINO, MoCo, SimCLR, SwAV, VICReg

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
            EarlyStopping(monitor=metric_name, patience=10, mode=metric_mode, verbose=True)
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
    train_config, model_config = config_builder(args)
    
    pl.seed_everything(train_config["seed"])
    torch.set_float32_matmul_precision('medium')
    
    backbone = backbone_builder(
        train_config["backbone"], 
        train_config["backbone_checkpoint"]
    )
    
    train_data, test_data, train_transform, test_transform = dataset_builder(
        train_config["ssl"], 
        train_config["dataset"], 
        train_config["data_root"], 
    )

    experiment_name = f'{train_config["ssl"]}_{train_config["backbone"]}_{train_config["dataset"]}'
    logger = WandbLogger(
        project="ssl-lightly",
        name=experiment_name,
        log_model=False,
        save_dir="."
    ) if train_config["wandb"] else TensorBoardLogger(
        save_dir="./tb_logs",
        name=experiment_name,
        default_hp_metric=False
    )
    
    def ssl_experiment():
        models = {
            "barlowtwins": BarlowTwins,
            "byol": BYOL,
            "dino": DINO,
            "moco": MoCo,
            "simclr": SimCLR,
            "swav": SwAV,
            "vicreg": VICReg,
        }
        
        model = models[train_config["ssl"]](backbone, train_config["batch_size"], **model_config)
        
        ssl_trainer = trainer_builder(
            train_config["devices"],
            f'./checkpoints/ssl/{train_config["ssl"]}/{train_config["backbone"]}/{train_config["dataset"]}', 
            logger,
            "train-ssl-loss",
            "min",
            train_config["ssl_epochs"]
        )
    
        train_data.transform = train_transform
        test_data.transform = test_transform

        train_loader = DataLoader(
            train_data, 
            batch_size=train_config["batch_size"], 
            shuffle=True, 
            num_workers=train_config["num_workers"], 
            pin_memory=True,
            generator=torch.Generator().manual_seed(train_config["seed"]),
        )
        
        valid_loader = DataLoader(
            test_data, 
            batch_size=train_config["batch_size"], 
            shuffle=False, 
            num_workers=train_config["num_workers"], 
            pin_memory=True,
            generator=torch.Generator().manual_seed(train_config["seed"]),
        )
        
        ssl_trainer.fit(
            model, 
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
        
    if "train" in train_config["experiment"]:
        ssl_experiment()

if __name__ == "__main__":
    import argparse

    from backbone import AVAILABLE_BACKBONES

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet18", choices=AVAILABLE_BACKBONES)
    parser.add_argument("--ssl", type=str, default="byol", choices=["barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"])
    parser.add_argument("--sl", type=str, default="linear", choices=["linear", "finetune"])
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()

    main(args)