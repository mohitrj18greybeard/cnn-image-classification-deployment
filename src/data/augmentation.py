"""
Data Augmentation Module (PyTorch Version)
========================================
Note: Augmentation is now handled within src/data/dataset.py using torchvision.transforms.
This file is kept for backward compatibility but is no longer used in the PyTorch pipeline.
"""

import logging

logger = logging.getLogger(__name__)

class DataAugmentor:
    def __init__(self, config: dict):
        pass
    
    def augment_batch(self, images, labels):
        return images, labels
