import os

from torchvision import datasets
from torchvision import transforms as T
import torch
from torch.utils.data import random_split, Dataset
# BarlowTwins uses BYOL augmentations.
from lightly.transforms.byol_transform import (
    BYOLTransform, 
    BYOLView1Transform, 
    BYOLView2Transform
)
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms.swav_transform import SwaVTransform
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.vicreg_transform import VICRegTransform

class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def transform_builder(SSL, input_size, normalize):
    SSL = SSL.lower()
    
    gaussian_blur = 0.0
    cj_strength = 0.5
    if "byol" in SSL or "barlow" in SSL:
        transform = BYOLTransform(
            BYOLView1Transform(
                input_size = input_size,
                normalize = normalize,
                gaussian_blur = gaussian_blur
            ),
            BYOLView2Transform(
                input_size = input_size,
                normalize = normalize,
                gaussian_blur = gaussian_blur
            ),
        )

    elif "dino" in SSL:
        transform = DINOTransform(
            global_crop_size = input_size,
            normalize = normalize,
            n_local_views = 0, 
            gaussian_blur = (gaussian_blur,) * 3
        )   
    
    elif "moco" in SSL:
        transform = MoCoV2Transform(
            input_size = input_size,
            normalize = normalize
        )     

    elif "swav" in SSL:
        transform = SwaVTransform(
            # TODO different crop sizes for different views
            crop_sizes = [input_size],
            crop_counts = [2],
            crop_min_scales = [0.14],
            cj_strength = cj_strength,
            gaussian_blur = gaussian_blur,
            normalize = normalize
        )

    elif "simclr" in SSL:
        transform = SimCLRTransform(
            input_size = input_size,
            normalize = normalize,
            cj_strength = cj_strength, 
            gaussian_blur = gaussian_blur
        )
        
    elif "vicreg" in SSL:
        transform = VICRegTransform(
            input_size = input_size,
            normalize = normalize,
            cj_strength = cj_strength, 
            gaussian_blur = gaussian_blur
        )
        
    return transform


def dataset_builder(
    SSL: str,
    dataset: str,
    data_root: str = ".",
    seed: int = 2023
):  
    match dataset:    
        case "cifar10":
            normalize = dict(
                mean = (0.4914, 0.4822, 0.4465),
                std = (0.247, 0.243, 0.261)
            )
            input_size = 32
            data = datasets.CIFAR10
            
        case "cifar100":
            normalize = dict(
                mean = (0.5071, 0.4867, 0.4408),
                std = (0.2675, 0.2565, 0.2761)
            )
            input_size = 32
            data = datasets.CIFAR100
        
        case "stl10":
            normalize = dict(
                mean = (0.4467, 0.4398, 0.4066),
                std = (0.2603, 0.2565, 0.2712)
            )
            input_size = 96
            data = datasets.STL10
            
        case "imagenet":
            normalize = dict(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
            )
            input_size = 224
            data = datasets.ImageNet
            
        case _:
            raise NotImplementedError
    
    ssl_transform = transform_builder(
        SSL, 
        input_size, 
        normalize
    )
    sl_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=normalize["mean"], std=normalize["std"])
    ]) 
    split_generator = torch.Generator().manual_seed(seed)
    
    if dataset in ("cifar10", "cifar100"):
        train_data = data(root = data_root, train = True, download = True)
        train_data_len = len(train_data)
        finetune_data_len = int(train_data_len * 0.1)
        train_data_len = train_data_len - finetune_data_len
        
        train_data, finetune_data = random_split(train_data, [train_data_len, finetune_data_len], generator=split_generator)
        
        train_data = SubsetWrapper(train_data)
        finetune_data = SubsetWrapper(finetune_data)
        
        test_data = data(root = data_root, train = False, download = True)
        
    elif dataset == "imagenet":
        sl_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=normalize["mean"], std=normalize["std"])
        ])
        train_data = data(root = os.path.join(data_root, "train"), split = 'train')
        train_data_len = len(train_data)
        finetune_data_len = int(train_data_len * 0.1)
        train_data_len = train_data_len - finetune_data_len
        
        train_data, finetune_data = random_split(train_data, [train_data_len, finetune_data_len], generator=split_generator)
        train_data = SubsetWrapper(train_data)
        finetune_data = SubsetWrapper(finetune_data)
        test_data = data(root = os.path.join(data_root, "val"), split = 'val')
        
    elif dataset == "stl10":
        # Use unlabeled split as train, train split as fine-tuning/linear probe set
        train_data = data(root = data_root, split = 'unlabeled', download = True)
        finetune_data = data(root = data_root, split = 'train', download = True)
        test_data = data(root = data_root, split = 'test', download = True)

    else:
        raise NotImplementedError
    
    
    return train_data, finetune_data, test_data, ssl_transform, sl_transform