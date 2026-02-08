"""Visualization utilities for evaluation and error analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from pathlib import Path
import seaborn as sns


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_dice: List[float],
    val_dice: List[float],
    save_path: Path
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_dice: List of training Dice scores per epoch
        val_dice: List of validation Dice scores per epoch
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Dice curves
    ax2.plot(epochs, train_dice, 'b-', label='Train Dice', linewidth=2)
    ax2.plot(epochs, val_dice, 'r-', label='Val Dice', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    num_samples: int = 8,
    save_path: Path = None,
    channel_to_show: int = 0
):
    """
    Visualize predictions alongside ground truth.
    
    Args:
        images: Image array (N, C, H, W)
        predictions: Prediction array (N, 1, H, W)
        ground_truths: Ground truth array (N, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
        channel_to_show: Which channel to display for multispectral images
    """
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Show one channel of the multispectral image
        img = images[i, channel_to_show, :, :]
        pred = predictions[i, 0, :, :]
        gt = ground_truths[i, 0, :, :]
        
        # Normalize image for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        
        # Plot image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Input (Channel {channel_to_show})', fontsize=11)
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(gt, cmap='Blues', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth', fontsize=11)
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction', fontsize=11)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_error_analysis(
    images: np.ndarray,
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    num_samples: int = 8,
    save_path: Path = None,
    channel_to_show: int = 0
):
    """
    Visualize error analysis (FP and FN separately).
    
    Args:
        images: Image array (N, C, H, W)
        predictions: Prediction array (N, 1, H, W)
        ground_truths: Ground truth array (N, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
        channel_to_show: Which channel to display
    """
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        img = images[i, channel_to_show, :, :]
        pred = (predictions[i, 0, :, :] > 0.5).astype(float)
        gt = ground_truths[i, 0, :, :]
        
        # Calculate errors
        false_positives = (pred > gt).astype(float)
        false_negatives = (gt > pred).astype(float)
        
        # Normalize image
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        
        # Plot image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Input', fontsize=10)
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(gt, cmap='Blues', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        
        # Plot false positives
        axes[i, 2].imshow(false_positives, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title('False Positives', fontsize=10)
        axes[i, 2].axis('off')
        
        # Plot false negatives
        axes[i, 3].imshow(false_negatives, cmap='Oranges', vmin=0, vmax=1)
        axes[i, 3].set_title('False Negatives', fontsize=10)
        axes[i, 3].axis('off')
    
    plt.suptitle('Error Analysis: False Positives and False Negatives', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics_comparison(
    results_dict: dict,
    save_path: Path
):
    """
    Plot comparison of metrics across models.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
        save_path: Path to save the plot
    """
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall']
    num_metrics = len(metrics_to_plot)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        model_names = list(results_dict.keys())
        values = [results_dict[name][metric] for name in model_names]
        
        bars = axes[idx].bar(model_names, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(model_names)])
        axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=13, fontweight='bold')
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    true_positives: int,
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
    save_path: Path
):
    """
    Plot confusion matrix.
    
    Args:
        true_positives: Number of true positives
        true_negatives: Number of true negatives
        false_positives: Number of false positives
        false_negatives: Number of false negatives
        save_path: Path to save plot
    """
    cm = np.array([[true_negatives, false_positives],
                   [false_negatives, true_positives]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
