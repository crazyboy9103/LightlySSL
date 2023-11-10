from pytorch_lightning.accelerators import find_usable_cuda_devices

def config_builder(args):
    train_config = dict(
        backbone = args.backbone,
        backbone_checkpoint = "",
        num_workers = 8,
        batch_size = 256,
        ssl_epochs = 400,
        seed = 2023,
        dataset = args.dataset, # "cifar10", "cifar100", "stl10", "imagenet",
        data_root = args.data_root, # "/media/research/C658FE8F58FE7E0D/datasets/imagenet",
        sl = args.sl, # "linear", "finetune"
        ssl = args.ssl, # "barlowtwins", "byol", "dino", "moco", "simclr", "swav", "vicreg"
        wandb = True,
        experiment = "train+eval", # "train", "eval", "train+eval"
        devices = find_usable_cuda_devices(args.num_gpus)
    )
    
    match train_config["dataset"]:
        case "cifar10":
            num_classes = 10
        
        case "cifar100":
            num_classes = 100
        
        case "stl10":
            num_classes = 10
        
        case "imagenet":
            num_classes = 1000
        
        case _:
            raise NotImplementedError(f"Dataset {train_config['dataset']} not implemented.")
        
    
    model_config = dict(
        linear_head_kwargs = dict(
            num_classes = num_classes, 
            label_smoothing = 0,
            k = 200,
        )
    )
    
    match train_config["ssl"]:
        case "barlowtwins":
            if train_config["dataset"] == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 8192,
                    output_dim = 8192,
                )

            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,
                    output_dim = 2048,
                )
                
        case "byol":
            if train_config["dataset"] == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 4096,
                    output_dim = 256,
                )
                model_config["prediction_head_kwargs"] = dict(
                    hidden_dim = 4096,
                    output_dim = 256,
                )
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 1024,
                    output_dim = 256,
                )
                model_config["prediction_head_kwargs"] = dict(
                    hidden_dim = 1024,
                    output_dim = 256,
                )
        
        case "dino":
            if train_config["dataset"] == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,
                    bottleneck_dim = 256,
                    output_dim = 65536,
                    batch_norm = False, 
                    freeze_last_layer = -1,
                    norm_last_layer = True
                )
                
                model_config["loss_kwargs"] = dict(
                    warmup_teacher_temp = 0.04,
                    teacher_temp = 0,
                    warmup_teacher_temp_epochs = 30,
                    student_temp = 0.1,
                    center_momentum = 0.9
                )
            
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,
                    bottleneck_dim = 256,
                    output_dim = 2048,
                    batch_norm = True, 
                    freeze_last_layer = -1,
                    norm_last_layer = True
                )
                
                model_config["loss_kwargs"] = dict(
                    warmup_teacher_temp = 0.04,
                    teacher_temp = 0,
                    warmup_teacher_temp_epochs = 30,
                    student_temp = 0.1,
                    center_momentum = 0.9
                )
        
        case "moco":
            if train_config["dataset"] == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,  
                    output_dim = 65536,
                )
                
                model_config["loss_kwargs"] = dict(
                    temperature = 0.1,
                    memory_bank_size = 4096
                )
            
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,  
                    output_dim = 2048,
                )
                
                model_config["loss_kwargs"] = dict(
                    temperature = 0.1,
                    memory_bank_size = 4096
                )

        case "simclr":
            if train_config["dataset"] == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,  
                    output_dim = 2048,
                )
            
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 512,  
                    output_dim = 128,
                )

        case "swav":
            if train_config["dataset"] == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,  
                    output_dim = 2048,
                )
                
                model_config["prototype_kwargs"] = dict(
                    n_prototypes = 3000
                )
            
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 512,  
                    output_dim = 128,
                )
                
                model_config["prototype_kwargs"] = dict(
                    n_prototypes = 3000
                )
        
        case "vicreg":
            if train_config["dataset"] == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 8192,
                    output_dim = 8192,
                    num_layers = 3,
                )
                
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 8192,
                    output_dim = 2048,
                    num_layers = 3,
                )
               
    return train_config, model_config
    
    
