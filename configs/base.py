import torch 

train_config = dict(
    backbone = "convnext_base",
    backbone_checkpoint = "",
    num_workers = 8,
    batch_size = 256,
    ssl_epochs = 100,
    sl_epochs = 100,
    seed = 2023,
    dataset = "cifar10",
    data_root = "./data",
    ssl_lr = 6e-2,
    sl_lr = 1e-4,
    sl = "linear", # "linear", "finetune"
    ssl = "barlowtwins", # "barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"
    wandb = True,
    experiment = "ssl+eval"
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