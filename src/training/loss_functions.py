"""Loss functions for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    The Dice coefficient is commonly used in medical image segmentation
    and is particularly useful for imbalanced datasets.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            pred: Predicted segmentation (B, 1, H, W)
            target: Ground truth segmentation (B, 1, H, W)
            
        Returns:
            Dice loss value
        """
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute intersection and union
        intersection = (pred * target).sum()
       
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # Return loss (1 - Dice)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE and Dice Loss.
    
    This combines Binary Cross Entropy and Dice Loss, which can
    provide better training stability and performance.
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        """
        Initialize Combined Loss.
        
        Args:
            bce_weight: Weight for BCE loss component
            dice_weight: Weight for Dice loss component
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted segmentation (B, 1, H, W)
            target: Ground truth segmentation (B, 1, H, W)
            
        Returns:
            Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function for creating loss functions.
    
    Args:
        loss_type: Type of loss ('dice', 'bce', or 'combined')
        **kwargs: Additional arguments for loss initialization
        
    Returns:
        Loss function module
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
