"""Data transformations and augmentations for water segmentation."""

import numpy as np
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AdvancedImageTransform:
    """Advanced augmentation pipeline for satellite imagery.
    
    Attributes:
        transform: Albumentations transform pipeline
    """
    
    def __init__(self, p: float = 0.5, train: bool = True):
        """
        Initialize transform pipeline.
        
        Args:
            p: Probability of applying each transformation
            train: Whether this is for training (applies augmentations)
        """
        if train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=p),
                A.VerticalFlip(p=p),
                A.RandomRotate90(p=p),
                # Additional augmentations can be added here
                # A.GaussianBlur(blur_limit=3, p=0.2),
                # A.RandomBrightnessContrast(p=0.2),
            ])
        else:
            # No augmentation for validation/test
            self.transform = A.Compose([])
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply transformations to image and mask.
        
        Args:
            image: Input image (H, W, C)
            mask: Input mask (H, W)
            
        Returns:
            Transformed image and mask
        """
        # Ensure float32 and copy to avoid in-place modifications
        image = image.astype(np.float32).copy()
        mask = mask.astype(np.float32).copy()
        
        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        
        return transformed['image'], transformed['mask']


def normalize_multispectral_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize multispectral image per-channel.
    
    Args:
        image: Multispectral image (H, W, C)
        
    Returns:
        Normalized image
    """
    normalized_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        mean = channel.mean()
        std = channel.std()
        # Avoid division by zero
        normalized_image[:, :, i] = (channel - mean) / (std + 1e-7)
    
    return normalized_image


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple:
        """
        Convert to tensors.
        
        Args:
            image: Image array (H, W, C)
            mask: Mask array (H, W)
            
        Returns:
            Tuple of image and mask tensors
        """
        import torch
        
        # Convert to tensor and permute to (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # Add channel dimension to mask
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image_tensor, mask_tensor
