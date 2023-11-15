import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.accelerators import find_usable_cuda_devices

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from dataset import dataset_builder, DataModule
from backbone import backbone_builder
from config import config_builder
from modules import BarlowTwins, BYOL, DINO, MoCo, SimCLR, SwAV, VICReg

def main(args):
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    
    model_config = config_builder(args)
    devices = find_usable_cuda_devices(args.num_gpus)
    metric_name = "valid/online-linear-accuracy"
    experiment_name = f'{args.ssl}_{args.backbone}_{args.dataset}_tune'
 

    def lr_objective(trial: optuna.trial.Trial) -> float:
        base_lr = model_config["optimizer_kwargs"]["base_lr"]
        base_lr = trial.suggest_float("base_lr", base_lr / 10, base_lr * 10, log=True)
        model_config["optimizer_kwargs"]["base_lr"] = base_lr
        
        if args.ssl == "barlowtwins":
            no_weight_decay_base_lr = model_config["optimizer_kwargs"]["no_weight_decay_base_lr"]            
            no_weight_decay_base_lr = trial.suggest_float("no_weight_decay_base_lr", no_weight_decay_base_lr / 10, no_weight_decay_base_lr * 10, log=True)
            model_config["optimizer_kwargs"]["no_weight_decay_base_lr"] = no_weight_decay_base_lr
        
        backbone = backbone_builder(
            args.backbone
        )
        
        train_data, test_data, train_transform, test_transform = dataset_builder(
            args.ssl, 
            args.dataset, 
            args.data_root, 
        )

        logger = WandbLogger(
            project="ssl-lightly-tune",
            name=experiment_name,
            log_model=False,
            save_dir="."
        ) if args.wandb else True
        
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
        
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=metric_name)
        trainer = pl.Trainer(
            logger=logger, 
            max_epochs=args.epochs,
            precision="16-mixed",
            benchmark=True,
            enable_checkpointing=False,
            # limit_val_batches=PERCENT_VALID_EXAMPLES,
            callbacks=[
                ModelSummary(max_depth=-1),
                LearningRateMonitor(logging_interval="step"),
                pruning_callback
            ],
            fast_dev_run=False,
            sync_batchnorm=len(devices) > 1,
            devices=devices,
            log_every_n_steps=1,
            strategy="ddp" if len(devices) > 1 else "auto", 
            num_sanity_val_steps=0,
            use_distributed_sampler = True,
        )

        train_data.transform = train_transform
        test_data.transform = test_transform

        data_loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        
        datamodule = DataModule(train_data, test_data, data_loader_kwargs)
        
        trainer.fit(
            model, 
            datamodule=datamodule
        )
        
        pruning_callback.check_pruned()
        return trainer.callback_metrics[metric_name].item()
    
    pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner() )

    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name="pl_ddp",
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(lr_objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    

    
if __name__ == "__main__":
    import argparse

    from backbone import AVAILABLE_BACKBONES

    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_root", type=str, default="./data")
    
    # model args
    parser.add_argument("--backbone", type=str, default="resnet18", choices=AVAILABLE_BACKBONES)
    # training args
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # pretrain args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ssl", type=str, default="byol", choices=["barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"])

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