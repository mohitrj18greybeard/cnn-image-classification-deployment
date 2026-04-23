"""Custom CNN with residual blocks for image classification."""
import torch, torch.nn as nn, torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
        ) if stride != 1 or c_in != c_out else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))

class CustomCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["models"]["custom_cnn"]
        f = cfg["filters"]
        self.stem = nn.Sequential(nn.Conv2d(3, f[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(f[0]), nn.ReLU(inplace=True))
        self.layer1 = ResBlock(f[0], f[0])
        self.layer2 = ResBlock(f[0], f[1], stride=2)
        self.layer3 = ResBlock(f[1], f[2], stride=2)
        self.layer4 = ResBlock(f[2], f[3], stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(f[3], 256), nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"] * 0.5),
            nn.Linear(256, cfg["num_classes"]))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(x)
