import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB


from dataset import dataset_builder, DataModule
from backbone import backbone_builder
from config import config_builder
from modules import BarlowTwins, BYOL, DINO, MoCo, SimCLR, SwAV, VICReg

def main(args):
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('medium')
    
    model_config = config_builder(args)
    metric_name = "train/ssl-loss" # "valid/online-linear-accuracy"
    metric_mode = "min" # "max"

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 16, "GPU": 4}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute=metric_name,
            checkpoint_score_order=metric_mode,
        ),
    )
    
    base_lr = model_config["optimizer_kwargs"]["base_lr"]
    
    search_space = {
        "base_lr": tune.loguniform(base_lr / 100, min(base_lr * 100, 1), base=10),
    }
    
    if args.ssl == "barlowtwins":
        no_weight_decay_base_lr = model_config["optimizer_kwargs"]["no_weight_decay_base_lr"]   
        search_space["no_weight_decay_base_lr"] = tune.loguniform(no_weight_decay_base_lr / 100, min(no_weight_decay_base_lr * 100, 1), base=10)
    
    def train_func(config):
        model_config["optimizer_kwargs"]["base_lr"] = config["base_lr"]
        
        if args.ssl == "barlowtwins":
            model_config["optimizer_kwargs"]["no_weight_decay_base_lr"] = config["no_weight_decay_base_lr"]
            
        backbone = backbone_builder(
            args.backbone
        )
        
        train_data, test_data, train_transform, test_transform = dataset_builder(
            args.ssl, 
            args.dataset, 
            args.data_root, 
        )
        
        
        experiment_name = f'{args.ssl}_{args.backbone}_{args.dataset}_lr-{config["base_lr"]}'
        
        logger = WandbLogger(
            project="ssl-lightly-tune-loss",
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
        
        trainer = pl.Trainer(
            logger=logger, 
            callbacks=[RayTrainReportCallback()],
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
            sync_batchnorm=True,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_steps=args.steps
        )

        train_data.transform = train_transform
        test_data.transform = test_transform

        data_loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            # generator=seeded_generator,
        )
        
        datamodule = DataModule(train_data, test_data, data_loader_kwargs)
        
        trainer = prepare_trainer(trainer)
        
        trainer.fit(
            model, 
            datamodule=datamodule
        )
    
    def tune_bohb(num_samples=10):
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=args.steps,
            reduction_factor=3
        )

        tuner = tune.Tuner(
            TorchTrainer(
                train_func,
                scaling_config=scaling_config,
                run_config=run_config,
            ),
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric=metric_name,
                mode=metric_mode,
                num_samples=num_samples,
                scheduler=scheduler,
                search_alg=TuneBOHB(max_concurrent=4),
            ),
        )
        return tuner.fit()
    
    return tune_bohb(num_samples=20)
    
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
    parser.add_argument("--steps", type=int, default=2000)
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
    
    result = main(args)
    print(result)