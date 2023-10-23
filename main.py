import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import dataset_builder
from backbone import backbone_builder
from modules import BarlowTwins, BYOL, DINO, MoCo, SimCLR, SwAV, VICReg
from configs import barlowtwins, byol, dino, moco, simclr, swav, vicreg
from configs.base import train_config, optimizer_config, eval_optimizer_config

from modules import EvalModule

def trainer_builder(checkpoint_path, project_name, experiment_name, metric_name, epochs):
    os.makedirs(checkpoint_path, exist_ok=True)
    
    logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        log_model=False,
        save_dir="."
    )  
    
    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=epochs,
        precision="32",
        benchmark=True,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor=metric_name, mode="max"),
        ],
        fast_dev_run = True
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
            "ssl-lightly",
            f'{train_config["ssl"]}_{train_config["backbone"]}_{train_config["dataset"]}',
            "valid-ssl-loss",
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
            generator=torch.Generator().manual_seed(train_config["seed"])
        )
        
        valid_loader = DataLoader(
            finetune_data, 
            batch_size=train_config["batch_size"], 
            shuffle=False, 
            num_workers=train_config["num_workers"], 
            pin_memory=True,
            generator=torch.Generator().manual_seed(train_config["seed"])
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
            "ssl-lightly",
            f'{train_config["sl"]}_{train_config["backbone"]}_{train_config["dataset"]}',
            "valid-accuracy",
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
            generator=torch.Generator().manual_seed(train_config["seed"])
        )
        
        valid_loader = DataLoader(
            test_data, 
            batch_size=train_config["batch_size"], 
            shuffle=False, 
            num_workers=train_config["num_workers"], 
            pin_memory=True,
            generator=torch.Generator().manual_seed(train_config["seed"])
        )
        
        sl_trainer.fit(
            eval_module, 
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
    
    if "ssl" in train_config["experiment"]:
        ssl_experiment()
    
    if "eval" in train_config["experiment"]:
        sl_experiment()   

if __name__ == "__main__":
    main()