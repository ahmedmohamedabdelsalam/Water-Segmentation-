"""Evaluation metrics for semantic segmentation."""

import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import time


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Dice score.
    
    Args:
        pred: Predicted segmentation (B, 1, H, W) or flattened
        target: Ground truth segmentation
        
    Returns:
        Dice score
    """
    pred_binary = (pred > 0.5).float().view(-1)
    target_binary = target.float().view(-1)
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-7)
    
    return dice.item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        
    Returns:
        IoU score
    """
    pred_binary = (pred > 0.5).float().view(-1)
    target_binary = target.float().view(-1)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    iou = intersection / (union + 1e-7)
    
    return iou.item()


def calculate_precision_recall(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate precision and recall.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        
    Returns:
        Tuple of (precision, recall)
    """
    pred_binary = (pred > 0.5).float().view(-1)
    target_binary = target.float().view(-1)
    
    true_positives = (pred_binary * target_binary).sum()
    predicted_positives = pred_binary.sum()
    actual_positives = target_binary.sum()
    
    precision = (true_positives / (predicted_positives + 1e-7)).item()
    recall = (true_positives / (actual_positives + 1e-7)).item()
    
    return precision, recall


def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        
    Returns:
        Dictionary of metrics
    """
    dice = calculate_dice_score(pred, target)
    iou = calculate_iou(pred, target)
    precision, recall = calculate_precision_recall(pred, target)
    
    # F1 score (should equal Dice score for binary segmentation)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def measure_inference_time(model: torch.nn.Module, input_tensor: torch.Tensor, 
                          warmup_runs: int = 10, measure_runs: int = 100) -> float:
    """
    Measure average inference time.
    
    Args:
        model: Model to benchmark
        input_tensor: Sample input tensor
        warmup_runs: Number of warmup runs
        measure_runs: Number of measurement runs
        
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(measure_runs):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / measure_runs * 1000
    
    return avg_time_ms


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model size metrics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb
    }


class ConfusionMatrixMetrics:
    """Calculate metrics from confusion matrix."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update confusion matrix with batch.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
        """
        pred_binary = (pred > 0.5).float().cpu().numpy().flatten()
        target_binary = target.float().cpu().numpy().flatten()
        
        self.true_positives += ((pred_binary == 1) & (target_binary == 1)).sum()
        self.true_negatives += ((pred_binary == 0) & (target_binary == 0)).sum()
        self.false_positives += ((pred_binary == 1) & (target_binary == 0)).sum()
        self.false_negatives += ((pred_binary == 0) & (target_binary == 1)).sum()
    
    def compute(self) -> Dict[str, float]:
        """
        Compute metrics from accumulated confusion matrix.
        
        Returns:
            Dictionary of metrics
        """
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        accuracy = (self.true_positives + self.true_negatives) / (
            self.true_positives + self.true_negatives + 
            self.false_positives + self.false_negatives + 1e-7
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }
