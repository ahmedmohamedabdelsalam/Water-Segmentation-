"""Evaluation package for water segmentation."""

from .metrics import (
    calculate_dice_score,
    calculate_iou,
    calculate_precision_recall,
    calculate_all_metrics,
    measure_inference_time,
    get_model_size
)
from .evaluator import Evaluator, compare_models
from .visualization import (
    plot_training_curves,
    visualize_predictions,
    visualize_error_analysis,
    plot_metrics_comparison,
    plot_confusion_matrix
)

__all__ = [
    'calculate_dice_score',
    'calculate_iou',
    'calculate_precision_recall',
    'calculate_all_metrics',
    'measure_inference_time',
    'get_model_size',
    'Evaluator',
    'compare_models',
    'plot_training_curves',
    'visualize_predictions',
    'visualize_error_analysis',
    'plot_metrics_comparison',
    'plot_confusion_matrix',
]
