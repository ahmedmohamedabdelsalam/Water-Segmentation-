"""Evaluation script for water segmentation models."""

import argparse
from pathlib import Path
import sys
import torch
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device, setup_logger
from src.data import create_dataloaders
from src.models import create_model
from src.evaluation import Evaluator, visualize_predictions, visualize_error_analysis, plot_confusion_matrix
from src.training import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate water segmentation model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-vis-samples',
        type=int,
        default=8,
        help='Number of samples to visualize'
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = Config.from_yaml(Path(args.config))
    set_seed(config.seed)
    device = get_device(config.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('evaluation', output_dir, 'evaluation.log')
    logger.info(f"Starting evaluation...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    
    # Create dataloaders (we only need test loader)
    print("Loading data...")
    _, _, test_loader = create_dataloaders(
        images_dir=config.data.images_dir,
        labels_dir=config.data.labels_dir,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        batch_size=config.training.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed
    )
    
    # Create model
    print(f"Loading model: {config.model.type}")
    model = create_model(
        model_type=config.model.type,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        init_features=config.model.init_features,
        dropout_rate=config.model.dropout_rate
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create evaluator
    evaluator = Evaluator(model, device)
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluator.evaluate(test_loader)
    
    # Benchmark speed
    print("Benchmarking inference speed...")
    speed_results = evaluator.benchmark_speed()
    results.update(speed_results)
    
    # Log results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for key, value in results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
        logger.info(f"{key}: {value}")
    print("="*80)
    
    # Save results to JSON
    results_json_path = output_dir / 'test_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_json_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_csv_path = output_dir / 'test_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to {results_csv_path}")
    
    # Get predictions for visualization
    print("\nGenerating visualizations...")
    images, preds, gts = evaluator.get_predictions(test_loader)
    
    # Visualize predictions
    vis_path = output_dir / 'predictions.png'
    visualize_predictions(images, preds, gts, num_samples=args.num_vis_samples, save_path=vis_path)
    logger.info(f"Predictions visualization saved to {vis_path}")
    
    # Visualize error analysis
    error_path = output_dir / 'error_analysis.png'
    visualize_error_analysis(images, preds, gts, num_samples=args.num_vis_samples, save_path=error_path)
    logger.info(f"Error analysis saved to {error_path}")
    
    # Plot confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(
        results['true_positives'],
        results['true_negatives'],
        results['false_positives'],
        results['false_negatives'],
        cm_path
    )
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
