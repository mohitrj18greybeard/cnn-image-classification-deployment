"""
Dataset Loading and Preprocessing Module (PyTorch Version)
========================================================
Handles CIFAR-10 dataset loading, transforms, and splitting.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Production-grade dataset loader for CIFAR-10 image classification using PyTorch."""

    CLASS_NAMES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # CIFAR-10 Normalization Stats
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2470, 0.2435, 0.2616]

    def __init__(self, config: dict):
        """
        Initialize DatasetLoader.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.image_size = config["dataset"]["image_size"]
        self.upscale_size = config["dataset"]["upscale_size"]
        self.batch_size = config["training"]["batch_size"]
        self.val_split = config["dataset"]["validation_split"]
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def get_transforms(self, augment=False, for_transfer=False):
        """Get train/val/test transforms."""
        size = self.upscale_size if for_transfer else self.image_size
        
        tf_list = []
        if augment:
            aug_cfg = self.config["augmentation"]
            tf_list.extend([
                transforms.RandomHorizontalFlip() if aug_cfg["horizontal_flip"] else None,
                transforms.RandomRotation(aug_cfg["rotation_range"]),
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)) if aug_cfg["zoom_range"] > 0 else None,
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
            tf_list = [t for t in tf_list if t is not None]
        else:
            tf_list.append(transforms.Resize((size, size)))

        tf_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)
        ])
        
        return transforms.Compose(tf_list)

    def load_data(self, for_transfer=False):
        """Load and split CIFAR-10 dataset."""
        logger.info(f"Loading CIFAR-10 (Transfer={for_transfer})...")

        train_tf = self.get_transforms(augment=True, for_transfer=for_transfer)
        val_tf = self.get_transforms(augment=False, for_transfer=for_transfer)
        test_tf = self.get_transforms(augment=False, for_transfer=for_transfer)

        full_train_ds = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=test_tf)

        # Split validation
        num_train = len(full_train_ds)
        indices = list(range(num_train))
        split = int(np.floor(self.val_split * num_train))
        np.random.seed(42)
        np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]

        # Use different transform for validation by re-wrapping subset
        train_ds = Subset(full_train_ds, train_idx)
        val_ds = Subset(datasets.CIFAR10(root=self.data_dir, train=True, transform=val_tf), val_idx)

        logger.info(f"Loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

        return train_ds, val_ds, test_ds

    def get_loaders(self, for_transfer=False):
        """Get PyTorch DataLoaders."""
        train_ds, val_ds, test_ds = self.load_data(for_transfer)
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader

    def get_class_names(self):
        return self.CLASS_NAMES

    def preprocess_image(self, image_np, for_transfer=False):
        """Preprocess single image for inference."""
        from PIL import Image
        img = Image.fromarray(image_np)
        tf = self.get_transforms(augment=False, for_transfer=for_transfer)
        return tf(img).unsqueeze(0)
