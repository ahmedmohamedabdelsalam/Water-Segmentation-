"""Models package for water segmentation."""

from .unet_baseline import BaselineUNet
from .unet_enhanced import EnhancedUNet


def create_model(model_type: str, **kwargs):
    """
    Factory function for creating models.
    
    Args:
        model_type: Type of model ('baseline_unet' or 'enhanced_unet')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
    """
    if model_type == 'baseline_unet':
        return BaselineUNet(**kwargs)
    elif model_type == 'enhanced_unet':
        return EnhancedUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    'BaselineUNet',
    'EnhancedUNet',
    'create_model',
]
