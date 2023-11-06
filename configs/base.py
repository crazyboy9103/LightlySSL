import argparse

import torch 
from pytorch_lightning.accelerators import find_usable_cuda_devices
from lightly.utils.scheduler import CosineWarmupScheduler

import scheduler as lr_scheduler
from backbone import AVAILABLE_BACKBONES

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, default="resnet50", choices=AVAILABLE_BACKBONES)
parser.add_argument("--ssl", type=str, default="barlowtwins", choices=["barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"])
parser.add_argument("--sl", type=str, default="linear", choices=["linear", "finetune"])
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--data_root", type=str, default="./data")
parser.add_argument("--num_gpus", type=int, default=4)
args = parser.parse_args()


train_config = dict(
    backbone = args.backbone,
    backbone_checkpoint = "",
    num_workers = 8,
    batch_size = 256,
    ssl_epochs = 400,
    sl_epochs = 100,
    seed = 2023,
    dataset = args.dataset, # "cifar10", "cifar100", "stl10", "imagenet",
    data_root = args.data_root, # "/media/research/C658FE8F58FE7E0D/datasets/imagenet",
    ssl_lr = 6e-2,
    sl_lr = 1e-4,
    sl = args.sl, # "linear", "finetune"
    ssl = args.ssl, # "barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"
    wandb = True,
    experiment = "train+eval", # "train", "eval", "train+eval"
    devices = find_usable_cuda_devices(args.num_gpus)
)

# This is from lightly benchmark script
BASE_BATCH_SIZE = 128
train_config["ssl_lr"] *= train_config["batch_size"] / BASE_BATCH_SIZE
train_config["sl_lr"] *= train_config["batch_size"] / BASE_BATCH_SIZE


optimizer_config = dict(
    optimizer = torch.optim.SGD,
    optimizer_kwargs = dict(
        lr = train_config["ssl_lr"],
        momentum = 0.9, 
        weight_decay = 5e-4,
    ),
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs = dict(
        T_max = train_config["ssl_epochs"],
    ),
)

eval_optimizer_config = dict(
    optimizer = torch.optim.SGD,
    optimizer_kwargs = dict(
        lr = train_config["sl_lr"],
        momentum = 0.9, 
        weight_decay = 5e-4,
    ),
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs = dict(
        T_max = train_config["sl_epochs"],
    ),
)