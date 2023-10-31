import torch 
from lightning.pytorch.accelerators import find_usable_cuda_devices

train_config = dict(
    backbone = "resnet50",
    backbone_checkpoint = "",
    num_workers = 8,
    batch_size = 256,
    ssl_epochs = 10,
    sl_epochs = 10,
    seed = 2023,
    dataset = "stl10", # "cifar10", "cifar100", "stl10", "imagenet",
    data_root = "./data", # "/media/research/C658FE8F58FE7E0D/datasets/imagenet",
    ssl_lr = 6e-2,
    sl_lr = 1e-4,
    sl = "linear", # "linear", "finetune"
    ssl = "barlowtwins", # "barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"
    wandb = False,
    experiment = "train+eval", # "train", "eval", "train+eval"
    devices = find_usable_cuda_devices(2)
)

optimizer_config = dict(
    optimizer = torch.optim.SGD,
    optimizer_kwargs = dict(
        lr = train_config["ssl_lr"]
    ),
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs = dict(
        T_max = train_config["ssl_epochs"],
        eta_min = train_config["ssl_lr"],
    ),
)

eval_optimizer_config = dict(
    optimizer = torch.optim.SGD,
    optimizer_kwargs = dict(
        lr = train_config["sl_lr"]
    ),
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs = dict(
        T_max = train_config["sl_epochs"],
        eta_min = train_config["sl_lr"],
    ),
)