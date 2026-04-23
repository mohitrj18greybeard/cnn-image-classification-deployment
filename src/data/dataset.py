"""Dataset loading with torchvision — supports CIFAR-10 and custom folders."""
import os, logging, numpy as np, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

log = logging.getLogger(__name__)

CIFAR_MEAN, CIFAR_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class DatasetLoader:
    CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

    def __init__(self, config):
        self.cfg = config
        self.img_size = config["dataset"]["image_size"]
        self.up_size = config["dataset"]["upscale_size"]
        self.bs = config["training"]["batch_size"]
        self.val_split = config["dataset"]["val_split"]
        os.makedirs(config["paths"]["data_dir"], exist_ok=True)

    def _transforms(self, train=False, transfer=False):
        size = self.up_size if transfer else self.img_size
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if transfer else (CIFAR_MEAN, CIFAR_STD)
        if train:
            aug = self.cfg["augmentation"]
            t = [transforms.Resize((size, size)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomRotation(aug["rotation_range"]),
                 transforms.ColorJitter(**aug["color_jitter"]),
                 transforms.RandomErasing(p=aug["random_erasing"]),
                 transforms.ToTensor(), transforms.Normalize(mean, std)]
            # RandomErasing must be after ToTensor
            t_ordered = t[:4] + [t[5], t[6], t[4]]
            return transforms.Compose(t_ordered)
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    def get_loaders(self, transfer=False, subset_size=None):
        train_tf = self._transforms(train=True, transfer=transfer)
        val_tf = self._transforms(train=False, transfer=transfer)
        data_dir = self.cfg["paths"]["data_dir"]

        full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)

        n = len(full_train)
        idx = np.random.RandomState(42).permutation(n)
        split = int(self.val_split * n)
        train_idx, val_idx = idx[split:], idx[:split]

        if subset_size:
            train_idx = train_idx[:subset_size]
            val_idx = val_idx[:min(subset_size // 5, len(val_idx))]

        train_ds = Subset(full_train, train_idx)
        val_ds = Subset(datasets.CIFAR10(data_dir, train=True, transform=val_tf), val_idx)

        if subset_size:
            test_ds = Subset(test_ds, range(min(subset_size // 2, len(test_ds))))

        nw = 0  # Windows-safe
        return (DataLoader(train_ds, self.bs, shuffle=True, num_workers=nw),
                DataLoader(val_ds, self.bs, num_workers=nw),
                DataLoader(test_ds, self.bs, num_workers=nw))

    def get_class_names(self):
        return self.CLASS_NAMES

    def preprocess_image(self, img_np, transfer=False):
        from PIL import Image
        img = Image.fromarray(img_np) if not isinstance(img_np, Image.Image) else img_np
        img = img.convert("RGB")
        tf = self._transforms(train=False, transfer=transfer)
        return tf(img).unsqueeze(0)
