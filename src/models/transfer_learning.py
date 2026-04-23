"""
Transfer Learning Module (PyTorch Version)
=========================================
Pre-trained models (EfficientNet, ResNet) fine-tuned for CIFAR-10.
"""

import torch
import torch.nn as nn
from torchvision import models
import logging

logger = logging.getLogger(__name__)


class TransferLearningModel(nn.Module):
    def __init__(self, config):
        super(TransferLearningModel, self).__init__()
        tl_cfg = config["transfer_learning"]
        model_name = tl_cfg["model_name"]
        num_classes = config["dataset"]["num_classes"]

        if "EfficientNet" in model_name:
            self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=tl_cfg["dropout_rate"], inplace=True),
                nn.Linear(in_features, num_classes),
            )
        elif "ResNet" in model_name:
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported in PyTorch version.")

        # Initially freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze head
        if "EfficientNet" in model_name:
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.fc.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self, num_layers=20):
        """Unfreeze last N layers for fine-tuning."""
        # For simplicity, unfreeze everything or just the top blocks
        # In PyTorch, we can unfreeze specific children
        for param in self.base_model.parameters():
            param.requires_grad = True
        logger.info("Fine-tuning enabled: Unfrozen all layers.")

    def forward(self, x):
        return self.base_model(x)
