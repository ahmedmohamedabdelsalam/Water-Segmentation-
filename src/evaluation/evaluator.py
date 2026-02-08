"""Evaluation utilities for model assessment."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

from .metrics import (
    calculate_all_metrics,
    measure_inference_time,
    get_model_size,
    ConfusionMatrixMetrics
)


class Evaluator:
    """
    Evaluator for comprehensive model assessment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on dataloader.
        
        Args:
            dataloader: Dataloader to evaluate on
            
        Returns:
            Dictionary of average metrics
        """
        cm_metrics = ConfusionMatrixMetrics()
        all_dice = []
        all_iou = []
        all_precision = []
        all_recall = []
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate metrics
                metrics = calculate_all_metrics(outputs, masks)
                all_dice.append(metrics['dice'])
                all_iou.append(metrics['iou'])
                all_precision.append(metrics['precision'])
                all_recall.append(metrics['recall'])
                
                # Update confusion matrix
                cm_metrics.update(outputs, masks)
        
        # Compute average metrics
        cm_results = cm_metrics.compute()
        
        results = {
            'dice': np.mean(all_dice),
            'iou': np.mean(all_iou),
            'precision': np.mean(all_precision),
            'recall': np.mean(all_recall),
            'accuracy': cm_results['accuracy'],
            'true_positives': cm_results['true_positives'],
            'true_negatives': cm_results['true_negatives'],
            'false_positives': cm_results['false_positives'],
            'false_negatives': cm_results['false_negatives']
        }
        
        return results
    
    def benchmark_speed(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 12, 256, 256),
        warmup_runs: int = 10,
        measure_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            input_shape: Input tensor shape (B, C, H, W)
            warmup_runs: Number of warmup runs
            measure_runs: Number of measurement runs
            
        Returns:
            Dictionary of speed metrics
        """
        dummy_input = torch.randn(input_shape)
        avg_time_ms = measure_inference_time(
            self.model,
            dummy_input,
            warmup_runs,
            measure_runs
        )
        
        return {
            'inference_time_ms': avg_time_ms,
            'fps': 1000.0 / avg_time_ms
        }
    
    def get_predictions(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Get predictions for visualization.
        
        Args:
            dataloader: Dataloader
            threshold: Threshold for binarization
            
        Returns:
            Lists of (images, predictions, ground_truths)
        """
        images_list = []
        preds_list = []
        masks_list = []
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Getting predictions"):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Convert to binary predictions
                preds = (outputs > threshold).float()
                
                # Move to CPU and convert to numpy
                images_np = images.cpu().numpy()
                preds_np = preds.cpu().numpy()
                masks_np = masks.numpy()
                
                images_list.append(images_np)
                preds_list.append(preds_np)
                masks_list.append(masks_np)
        
        # Concatenate all batches
        images_all = np.concatenate(images_list, axis=0)
        preds_all = np.concatenate(preds_list, axis=0)
        masks_all = np.concatenate(masks_list, axis=0)
        
        return images_all, preds_all, masks_all


def compare_models(
    models_dict: Dict[str, nn.Module],
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models.
    
    Args:
        models_dict: Dictionary mapping model names to models
        dataloader: Dataloader for evaluation
        device: Device to run on
        
    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        evaluator = Evaluator(model, device)
        
        # Evaluate
        eval_metrics = evaluator.evaluate(dataloader)
        
        # Benchmark speed
        speed_metrics = evaluator.benchmark_speed()
        
        # Get model size
        size_metrics = get_model_size(model)
        
        # Combine all metrics
        results[model_name] = {
            **eval_metrics,
            **speed_metrics,
            **size_metrics
        }
    
    return results
