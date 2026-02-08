"""Training package for water segmentation."""

from .loss_functions import DiceLoss, CombinedLoss, get_loss_function
from .trainer import Trainer
from .config import Config

__all__ = [
    'DiceLoss',
    'CombinedLoss',
    'get_loss_function',
    'Trainer',
    'Config',
]
