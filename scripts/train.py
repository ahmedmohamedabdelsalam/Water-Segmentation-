"""Training script for water segmentation models."""

import argparse
from pathlib import Path
import sys
import torch
import torch.optim as optim

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, get_device
from src.data import create_dataloaders
from src.models import create_model
from src.training import get_loss_function, Trainer, Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train water segmentation model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode with small subset'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = Config.from_yaml(config_path)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create experiment name
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config.model.type}_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    print(f"Experiment name: {experiment_name}")
    
    # Create dataloaders
    print(f"\nLoading data from:")
    print(f"  Images: {config.data.images_dir}")
    print(f"  Labels: {config.data.labels_dir}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        images_dir=config.data.images_dir,
        labels_dir=config.data.labels_dir,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        batch_size=config.training.batch_size if not args.quick_test else 2,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        augmentation_prob=config.augmentation.train['horizontal_flip']
    )
    
    # Create model
    print(f"\nCreating model: {config.model.type}")
    model = create_model(
        model_type=config.model.type,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        init_features=config.model.init_features,
        dropout_rate=config.model.dropout_rate
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    if config.training.loss_type == 'combined':
        criterion = get_loss_function(
            config.training.loss_type,
            bce_weight=config.training.loss_weights['bce'],
            dice_weight=config.training.loss_weights['dice']
        )
    else:
        criterion = get_loss_function(config.training.loss_type)
    
    print(f"\nLoss function: {config.training.loss_type}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.training.lr_scheduler.mode,
        factor=config.training.lr_scheduler.factor,
        patience=config.training.lr_scheduler.patience,
        min_lr=config.training.lr_scheduler.min_lr
    )
    
    # Create trainer
    experiment_dir = Path(config.logging.save_dir)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        gradient_clip_max_norm=config.training.gradient_clip_max_norm,
        log_interval=config.logging.log_interval,
        save_checkpoints=config.logging.save_checkpoints,
        checkpoint_interval=config.logging.checkpoint_interval,
        use_tensorboard=config.logging.tensorboard
    )
    
    # Train
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
    print("="*80)
    
    num_epochs = 2 if args.quick_test else config.training.num_epochs
    
    trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=config.training.early_stopping.patience,
        early_stopping_min_delta=config.training.early_stopping.min_delta,
        early_stopping_metric=config.training.early_stopping.metric
    )
    
    print("="*80)
    print(f"Training complete! Best model saved at epoch {trainer.best_epoch}")
    print(f"Results saved to: {trainer.experiment_dir}")


if __name__ == '__main__':
    main()
