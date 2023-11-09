import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from dataset import dataset_builder
from backbone import backbone_builder
from configs.base import train_config, optimizer_config, eval_optimizer_config

match train_config["dataset"]:
    case "cifar10":
        from configs.cifar10 import barlowtwins, byol, dino, moco, simclr, swav, vicreg
    case "cifar100":
        from configs.cifar100 import barlowtwins, byol, dino, moco, simclr, swav, vicreg
    case "stl10":
        from configs.stl10 import barlowtwins, byol, dino, moco, simclr, swav, vicreg
    case "imagenet":
        from configs.imagenet import barlowtwins, byol, dino, moco, simclr, swav, vicreg
    case _:
        raise NotImplementedError
    
from modules import BarlowTwins, BYOL, DINO, MoCo, SimCLR, SwAV, VICReg

def trainer_builder(
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
        # deterministic=True,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor=metric_name, mode=metric_mode),
            ModelSummary(max_depth=-1),
            EarlyStopping(monitor=metric_name, patience=10, mode=metric_mode, verbose=True)
        ],
        fast_dev_run = False,
        sync_batchnorm=True,
        devices=train_config["devices"],
        log_every_n_steps=1,
        strategy="auto", # "ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        use_distributed_sampler = True,
    )
    return trainer

def main():  
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
        train_config["seed"]
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
        config = optimizer_config
        model_configs = {
            "barlowtwins": barlowtwins.model_config,
            "byol": byol.model_config,
            "dino": dino.model_config,
            "moco": moco.model_config,
            "simclr": simclr.model_config,
            "swav": swav.model_config,
            "vicreg": vicreg.model_config,
        }
        
        models = {
            "barlowtwins": BarlowTwins,
            "byol": BYOL,
            "dino": DINO,
            "moco": MoCo,
            "simclr": SimCLR,
            "swav": SwAV,
            "vicreg": VICReg,
        }
        
        ssl = train_config["ssl"]
        config.update(model_configs[ssl])
        model = models[ssl](backbone, **config)
        
        ssl_trainer = trainer_builder(
            f'./checkpoints/ssl/{train_config["ssl"]}/{train_config["backbone"]}/{train_config["dataset"]}', 
            logger,
            "valid-ssl-loss",
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
            drop_last=True
        )
        
        valid_loader = DataLoader(
            test_data, 
            batch_size=train_config["batch_size"], 
            shuffle=False, 
            num_workers=train_config["num_workers"], 
            pin_memory=True,
            generator=torch.Generator().manual_seed(train_config["seed"]),
            drop_last=True
        )
        
        ssl_trainer.fit(
            model, 
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
        
        
    if "train" in train_config["experiment"]:
        ssl_experiment()

if __name__ == "__main__":
    main()