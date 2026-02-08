"""Reproducibility utilities for ensuring deterministic training."""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_name: str = "auto") -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_name: Device name ('cuda', 'cpu', or 'auto')
        
    Returns:
        torch.device object
    """
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    return device
