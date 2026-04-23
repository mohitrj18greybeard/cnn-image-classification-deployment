"""Utility functions: config loading, seeding, logging."""
import os, yaml, random, logging, torch
import numpy as np

def load_config(path: str = "config/config.yaml") -> dict:
    if not os.path.isabs(path) and not os.path.exists(path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(level="INFO"):
    logging.basicConfig(level=getattr(logging, level),
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
