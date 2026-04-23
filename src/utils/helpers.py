"""
Utility Functions
=================
Configuration loading, logging setup, reproducibility, and helper functions.
"""

import os
import yaml
import logging
import random
import numpy as np
import torch


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    # Handle both absolute and relative paths
    if not os.path.isabs(config_path):
        # Try relative to current working directory first
        if not os.path.exists(config_path):
            # Try relative to the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            config_path = os.path.join(project_root, config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the project.

    Args:
        level: Logging level string.
    """
    log_format = (
        "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model_size_mb(model) -> float:
    """
    Calculate model size in megabytes.

    Args:
        model: Keras model.

    Returns:
        Model size in MB.
    """
    total_params = model.count_params()
    # Assuming float32 (4 bytes per parameter)
    size_mb = (total_params * 4) / (1024 * 1024)
    return round(size_mb, 2)


def format_metrics(metrics: dict) -> str:
    """
    Format evaluation metrics as a readable string.

    Args:
        metrics: Dictionary of metric names and values.

    Returns:
        Formatted string.
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:.<30s} {value:.4f}")
        elif isinstance(value, (int, np.integer)):
            lines.append(f"  {key:.<30s} {value:,}")
        elif isinstance(value, str):
            lines.append(f"  {key:.<30s} {value}")
    return "\n".join(lines)


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        The path string.
    """
    os.makedirs(path, exist_ok=True)
    return path
