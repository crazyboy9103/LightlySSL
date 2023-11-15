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
        ),
        online_knn_head_kwargs = dict(
            num_classes = num_classes, 
            k = args.k
        ),
    )
    
    match args.ssl:
        case "barlowtwins":
            if args.dataset == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 8192,
                    output_dim = 8192,
                )
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.2, 
                    no_weight_decay_base_lr=0.0048
                )

            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 2048,
                    output_dim = 2048,
                )
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.06, 
                    no_weight_decay_base_lr=0.06
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
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.06, 
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
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.45, 
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
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.03
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
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.06
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
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.03
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
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.06
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
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.3
                )
            
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 512,  
                    output_dim = 128,
                )
                model_config["loss_kwargs"] = dict(
                    temperature = 0.5,
                )
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.06
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
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.6
                )
            
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 512,  
                    output_dim = 128,
                )
                model_config["prototype_kwargs"] = dict(
                    n_prototypes = 3000
                )
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.001
                )
        
        case "vicreg":
            if args.dataset == "imagenet":
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 8192,
                    output_dim = 8192,
                    num_layers = 3,
                )
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.8
                )
                
            else:
                model_config["projection_head_kwargs"] = dict(
                    hidden_dim = 8192,
                    output_dim = 2048,
                    num_layers = 3,
                )
                model_config["optimizer_kwargs"] = dict(
                    base_lr=0.8
                )
                
        case _:
            raise NotImplementedError(f"SSL {args.ssl} not implemented.")
               
    return model_config
    
    
