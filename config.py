from pytorch_lightning.accelerators import find_usable_cuda_devices

def config_builder(args):
    match args.dataset:
        case "cifar10":
            num_classes = 10
        
        case "cifar100":
            num_classes = 100
        
        case "stl10":
            num_classes = 10
        
        case "imagenet":
            num_classes = 1000
        
        case _:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
        
    
    model_config = dict(
        online_linear_head_kwargs = dict(
            num_classes = num_classes, 
            label_smoothing = args.label_smoothing,
        )
    )
    
    match args.ssl:
        case "barlowtwins":
            if args.dataset == "imagenet":
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
            if args.dataset == "imagenet":
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
            if args.dataset == "imagenet":
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
            if args.dataset == "imagenet":
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
            if args.dataset == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,  
                    output_dim = 2048,
                )
                
                model_config["loss_kwargs"] = dict(
                    temperature = 0.1,
                )
            
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 512,  
                    output_dim = 128,
                )
                
                model_config["loss_kwargs"] = dict(
                    temperature = 0.5,
                )

        case "swav":
            if args.dataset == "imagenet":
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
            if args.dataset == "imagenet":
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
        case _:
            raise NotImplementedError(f"SSL {args.ssl} not implemented.")
               
    return model_config
    
    
