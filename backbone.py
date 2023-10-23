import torch
from torch import nn
import timm

AVAILABLE_BACKBONES = [
    "convnext_large",
    "convnext_base",
    "convnext_tiny",

    "resnet18",
    "resnet50",

    "resnext50_32x4d",

    # "vit_tiny_patch16_224",
    # "vit_tiny_patch16_384",
    # "vit_small_patch16_224",
    # "vit_small_patch16_384",
    # "vit_large_patch16_224",
    # "vit_large_patch16_384",
]

def backbone_builder(backbone_name, checkpoint_path=""):
    assert backbone_name in AVAILABLE_BACKBONES
    backbone = timm.create_model(backbone_name)
    
    if "convnext" in backbone_name:
        backbone.head = nn.Sequential(*tuple(backbone.head.children())[:3])
    
    elif "resnet" in backbone_name or "resnext" in backbone_name:
        backbone.fc = nn.Identity()
    
    # elif "vit" in backbone_name:
    #     backbone.head_drop = nn.Identity()
    #     backbone.head = nn.Identity()

    if checkpoint_path:
        backbone.load_state_dict(torch.load(checkpoint_path))
    
    input_size = 224
    # if "224" in backbone_name:
    #     input_size = 224
    # elif "384" in backbone_name:
    #     input_size = 384
        
    backbone.eval()
    with torch.no_grad():
        output = backbone(torch.randn(1, 3, input_size, input_size))
    
    # Set the output dimension of the backbone, which is needed for subsequent heads
    output_dim = output.shape[1]
    if hasattr(backbone, "output_dim"):
        raise ValueError("The backbone already has an attribute 'output_dim'")
        
    setattr(backbone, "output_dim", output_dim)
    backbone.train() # By default, backbone is in train mode
    return backbone