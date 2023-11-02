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
from configs import barlowtwins, byol, dino, moco, simclr, swav, vicreg
from modules import BarlowTwins, BYOL, DINO, MoCo, SimCLR, SwAV, VICReg
from modules import EvalModule

def trainer_builder(checkpoint_path, logger, metric_name, metric_mode, epochs):
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
    
    train_data, finetune_data, test_data, ssl_transform, sl_transform = dataset_builder(
        train_config["ssl"], 
        train_config["dataset"], 
        train_config["data_root"], 
        train_config["seed"]
    )

    logger = WandbLogger(
        project="ssl-lightly",
        name=f'{train_config["ssl"]}_{train_config["backbone"]}_{train_config["dataset"]}',
        log_model=False,
        save_dir="."
    ) if train_config["wandb"] else TensorBoardLogger(
        save_dir="./tb_logs",
        name=f'{train_config["ssl"]}_{train_config["backbone"]}_{train_config["dataset"]}',
        default_hp_metric=False
    )
    
    def ssl_experiment():
        config = optimizer_config
        match train_config["ssl"]:
            case "barlowtwins":
                config.update(barlowtwins.model_config)
                model = BarlowTwins(backbone, **config)
            
            case "byol":
                config.update(byol.model_config)
                model = BYOL(backbone, **config)
            
            case "dino":
                config.update(dino.model_config)
                model = DINO(backbone, **config)
            
            case "moco":
                config.update(moco.model_config)
                model = MoCo(backbone, **config)
            
            case "simclr":
                config.update(simclr.model_config)
                model = SimCLR(backbone, **config)
            
            case "swav":
                config.update(swav.model_config)
                model = SwAV(backbone, **config)
            
            case "vicreg":
                config.update(vicreg.model_config)
                model = VICReg(backbone, **config)
            
            case _:
                raise NotImplementedError
        
        ssl_trainer = trainer_builder(
            f'./checkpoints/ssl/{train_config["ssl"]}/{train_config["backbone"]}/{train_config["dataset"]}', 
            logger,
            "valid-ssl-loss",
            "min",
            train_config["ssl_epochs"]
        )
    
        train_data.transform = ssl_transform
        finetune_data.transform = ssl_transform

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
            finetune_data, 
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
    
    def sl_experiment():
        eval_module = EvalModule(
            backbone, 
            eval_type = train_config["sl"],
            num_classes = len(test_data.classes),
            **eval_optimizer_config
        )
        
        sl_trainer = trainer_builder(
            f'./checkpoints/sl/{train_config["sl"]}/{train_config["backbone"]}/{train_config["dataset"]}', 
            logger,
            "valid-accuracy",
            "max",
            train_config["sl_epochs"]
        )
        
        finetune_data.transform = sl_transform
        test_data.transform = sl_transform
        
        train_loader = DataLoader(
            finetune_data, 
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
        
        sl_trainer.fit(
            eval_module, 
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
    
    if "train" in train_config["experiment"]:
        ssl_experiment()
    
    if "eval" in train_config["experiment"]:
        sl_experiment()   

if __name__ == "__main__":
    main()