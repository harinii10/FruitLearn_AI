import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as T2

def check_corrupt_images(image_paths):
    """Scans image list for corrupt or unreadable files."""
    valid_paths = []
    corrupt_count = 0
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
        except Exception:
            corrupt_count += 1
    return valid_paths, corrupt_count

def get_transforms(image_size=256, is_training=True):
    """Fast minimal CPU transform: resize and convert to float tensor."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])

class GPUAugmenter(torch.nn.Module):
    """Executes 100% of data augmentations directly on GPU CUDA tensors."""
    def __init__(self, image_size=256, is_training=True):
        super().__init__()
        self.is_training = is_training
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if is_training:
            self.aug = T2.Compose([
                T2.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
                T2.RandomHorizontalFlip(p=0.5),
                T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T2.Normalize(mean=mean, std=std)
            ])
        else:
            self.aug = T2.Normalize(mean=mean, std=std)

    def forward(self, x):
        return self.aug(x)

class GPUDINOv2Augmenter(torch.nn.Module):
    """Executes DINOv2 multi-crop views directly on GPU CUDA tensors."""
    def __init__(self, image_size=256):
        super().__init__()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.aug = T2.Compose([
            T2.RandomResizedCrop(image_size, scale=(0.5, 1.0), antialias=True),
            T2.RandomHorizontalFlip(p=0.5),
            T2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T2.Normalize(mean=mean, std=std)
        ])

    def forward(self, x):
        view1 = self.aug(x)
        view2 = self.aug(x)
        return view1, view2

class DINOv2Augmentation:
    """Multi-view augmentation wrapper for compatibility."""
    def __init__(self, image_size=256):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.global_trans = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def __call__(self, image):
        view1 = self.global_trans(image)
        view2 = self.global_trans(image)
        return view1, view2
