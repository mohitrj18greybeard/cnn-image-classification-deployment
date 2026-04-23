"""EfficientNet transfer learning model."""
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["models"]["efficientnet"]
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(in_feat, cfg["num_classes"]))
        if cfg["freeze_backbone"]:
            for p in list(self.backbone.parameters())[:-cfg["fine_tune_layers"]]:
                p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
