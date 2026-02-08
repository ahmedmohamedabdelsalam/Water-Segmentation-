"""Run automated experiments comparing different configurations."""

import argparse
from pathlib import Path
import sys
import subprocess
import json
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import plot_metrics_comparison


def run_experiment(config_path: str, experiment_name: str):
    """
    Run a single experiment.
    
    Args:
        config_path: Path to config file
        experiment_name: Name for the experiment
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Run training
    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", config_path,
        "--experiment-name", experiment_name
    ]
    
    result = subprocess.run(train_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error running experiment {experiment_name}")
        return None
    
    # Find best checkpoint
    checkpoint_dir = Path("experiments") / experiment_name / "checkpoints"
    best_checkpoint = checkpoint_dir / "best_model.pth"
    
    if not best_checkpoint.exists():
        print(f"Best checkpoint not found for {experiment_name}")
        return None
    
    # Run evaluation
    eval_output_dir = Path("experiments") / experiment_name / "evaluation"
    eval_cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--config", config_path,
        "--checkpoint", str(best_checkpoint),
        "--output-dir", str(eval_output_dir)
    ]
    
    result = subprocess.run(eval_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error evaluating experiment {experiment_name}")
        return None
    
    # Load results
    results_path = eval_output_dir / "test_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_experiment_configs():
    """Create experiment configuration files."""
    configs_dir = Path("configs/experiments")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = {}
    
    # Experiment 1: Baseline U-Net with Dice Loss
    exp1_config = """
data:
  images_dir: "data/data/images"
  labels_dir: "data/data/labels"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  type: "baseline_unet"
  in_channels: 12
  out_channels: 1
  init_features: 64
  dropout_rate: 0.0

training:
  batch_size: 8
  num_epochs: 50
  learning_rate: 0.0005
  weight_decay: 0.0001
  gradient_clip_max_norm: 0.5
  loss_type: "dice"
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "min"
    factor: 0.5
    patience: 10
    min_lr: 0.00001
  early_stopping:
    patience: 15
    min_delta: 0.001
    metric: "val_dice"

augmentation:
  train:
    enable: true
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation_90: 0.5
  val:
    enable: false
  test:
    enable: false

seed: 42
device: "auto"
num_workers: 4
pin_memory: true

logging:
  log_interval: 10
  save_dir: "experiments"
  tensorboard: true
  save_checkpoints: true
  checkpoint_interval: 5
"""
    exp1_path = configs_dir / "baseline_dice.yaml"
    with open(exp1_path, 'w') as f:
        f.write(exp1_config)
    experiments['baseline_dice'] = str(exp1_path)
    
    # Experiment 2: Enhanced U-Net with Combined Loss
    exp2_config = exp1_config.replace('type: "baseline_unet"', 'type: "enhanced_unet"')
    exp2_config = exp2_config.replace('loss_type: "dice"', 'loss_type: "combined"')
    exp2_config = exp2_config.replace('dropout_rate: 0.0', 'dropout_rate: 0.3')
    exp2_path = configs_dir / "enhanced_combined.yaml"
    with open(exp2_path, 'w') as f:
        f.write(exp2_config)
    experiments['enhanced_combined'] = str(exp2_path)
    
    # Experiment 3: Enhanced U-Net without augmentation  
    exp3_config = exp2_config.replace('enable: true', 'enable: false')
    exp3_path = configs_dir / "enhanced_no_aug.yaml"
    with open(exp3_path, 'w') as f:
        f.write(exp3_config)
    experiments['enhanced_no_aug'] = str(exp3_path)
    
    return experiments


def main():
    """Run all experiments and compare results."""
    parser = argparse.ArgumentParser(description='Run automated experiments')
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=None,
        help='List of experiments to run (default: all)'
    )
    args = parser.parse_args()
    
    print("Creating experiment configurations...")
    experiments = create_experiment_configs()
    
    # Select experiments to run
    if args.experiments:
        experiments = {k: v for k, v in experiments.items() if k in args.experiments}
    
    # Run experiments
    results = {}
    for exp_name, config_path in experiments.items():
        exp_results = run_experiment(config_path, exp_name)
        if exp_results:
            results[exp_name] = exp_results
    
    # Create comparison
    if len(results) > 0:
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPARISON")
        print(f"{'='*80}\n")
        
        # Create comparison table
        comparison_df = pd.DataFrame(results).T
        print(comparison_df[[
            'dice', 'iou', 'precision', 'recall',
            'inference_time_ms', 'total_params', 'size_mb'
        ]].to_string())
        
        # Save comparison
        comparison_dir = Path("experiments/comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_csv = comparison_dir / f"comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_csv)
        print(f"\nComparison saved to: {comparison_csv}")
        
        # Plot comparison
        plot_path = comparison_dir / f"comparison_{timestamp}.png"
        plot_metrics_comparison(results, plot_path)
        print(f"Comparison plot saved to: {plot_path}")
        
        print(f"\n{'='*80}")
        print("All experiments complete!")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
