import os

from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import Dataset
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

INPUT_SIZES = {
    "cifar10": 32,
    "cifar100": 32,
    "stl10": 96,
    "imagenet": 224
}

STATS = {
    "cifar10": dict(
        mean = (0.4914, 0.4822, 0.4465),
        std = (0.247, 0.243, 0.261)
    ),
    "cifar100": dict(
        mean = (0.5071, 0.4867, 0.4408),
        std = (0.2675, 0.2565, 0.2761)
    ),
    "stl10": dict(
        mean = (0.4467, 0.4398, 0.4066),
        std = (0.2603, 0.2565, 0.2712)
    ),
    "imagenet": dict(
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)
    )
}

class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.dataset)

def transform_builder(
    ssl: str,
    dataset: str
):  
    input_size = INPUT_SIZES[dataset]
    norm_stats = STATS[dataset]
    if dataset == "imagenet":
        test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=norm_stats["mean"], std=norm_stats["std"])
        ])
        match ssl:
            case "barlowtwins":
                train_transform = BYOLTransform(
                    BYOLView1Transform(
                        input_size = input_size,
                        normalize = norm_stats,
                    ),
                    BYOLView2Transform(
                        input_size = input_size,
                        normalize = norm_stats,
                    ),
                )
            
            case "byol":
                train_transform = BYOLTransform(
                    BYOLView1Transform(
                        input_size = input_size,
                        normalize = norm_stats,
                    ),
                    BYOLView2Transform(
                        input_size = input_size,
                        normalize = norm_stats,
                    ),
                )
            
            
            case "dino":
                train_transform = DINOTransform(
                    global_crop_size = input_size,
                    normalize = norm_stats,
                    global_crop_scale = (0.14, 1),
                    local_crop_scale = (0.05, 0.14)
                )
                
            case "moco":
                train_transform = SimCLRTransform(
                    input_size = input_size,
                    normalize = norm_stats,
                )
            
            case "simclr":
                train_transform = SimCLRTransform(
                    input_size = input_size,
                    normalize = norm_stats,
                )
            
            case "swav":
                train_transform = SwaVTransform(
                    crop_sizes = (input_size, 96),
                    crop_counts = (2, 6),
                    normalize = norm_stats,
                )
            
            case "vicreg":
                train_transform = VICRegTransform(
                    input_size = input_size,
                    normalize = norm_stats,
                )
            
            case _:
                raise NotImplementedError
                
       
    elif dataset in ("cifar10", "cifar100", "stl10"):
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=norm_stats["mean"], std=norm_stats["std"])
        ]) 
        match ssl:
            case "barlowtwins":
                train_transform = BYOLTransform(
                    BYOLView1Transform(
                        input_size = input_size,
                        normalize = norm_stats,
                        gaussian_blur = 0
                    ),
                    BYOLView2Transform(
                        input_size = input_size,
                        normalize = norm_stats,
                        gaussian_blur = 0
                    ),
                )
            
            case "byol":
                train_transform = BYOLTransform(
                    BYOLView1Transform(
                        input_size = input_size,
                        normalize = norm_stats,
                        gaussian_blur = 0
                    ),
                    BYOLView2Transform(
                        input_size = input_size,
                        normalize = norm_stats,
                        gaussian_blur = 0
                    ),
                )
            
            case "dino":
                train_transform = DINOTransform(
                    global_crop_size = input_size,
                    normalize = norm_stats,
                    cj_strength = 0.5,
                    n_local_views = 0, 
                    gaussian_blur = (0,) * 3
                )
            
            case "moco":
                train_transform = SimCLRTransform(
                    input_size = input_size,
                    normalize = norm_stats,
                    gaussian_blur = 0
                )
            
            case "simclr":
                train_transform = SimCLRTransform(
                    input_size = input_size,
                    normalize = norm_stats,
                    gaussian_blur = 0
                )

            case "swav":
                train_transform = SwaVTransform(
                    crop_sizes = [input_size,],
                    crop_counts = 2,
                    crop_min_scales = [0.14,],
                    cj_strength = 0.5, 
                    gaussian_blur = 0
                )
            
            case "vicreg":
                train_transform = VICRegTransform(
                    input_size = input_size,
                    normalize = norm_stats,
                    cj_strength = 0.5,
                    gaussian_blur = 0
                )
            
            case _:
                raise NotImplementedError
    
    else:
        raise NotImplementedError
        
    return train_transform, test_transform 


def dataset_builder(
    ssl: str,
    dataset: str,
    data_root: str = ".",
):  
    train_transform, test_transform  = transform_builder(ssl, dataset)
    
    data = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "stl10": datasets.STL10,
        "imagenet": datasets.ImageNet
    }[dataset]
    
    if dataset in ("cifar10", "cifar100"):
        train_data = data(root = data_root, train = True, download = True)
        test_data = data(root = data_root, train = False, download = True)
        
    elif dataset == "imagenet":
        train_data = data(root = os.path.join(data_root, "train"), split = 'train')
        test_data = data(root = os.path.join(data_root, "val"), split = 'val')
        
    elif dataset == "stl10":
        # TODO 
        # train_data = data(root = data_root, split = 'unlabeled', download = True)
        # finetune_data = data(root = data_root, split = 'train', download = True)
        # test_data = data(root = data_root, split = 'test', download = True)
        raise NotImplementedError

    else:
        raise NotImplementedError
    
    train_data = DatasetWrapper(train_data)
    test_data = DatasetWrapper(test_data)
    return train_data, test_data, train_transform, test_transform