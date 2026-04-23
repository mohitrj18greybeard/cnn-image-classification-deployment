"""Model architectures module."""
from src.models.custom_cnn import CustomCNN
from src.models.transfer_learning import TransferLearningModel

__all__ = ["CustomCNN", "TransferLearningModel"]
