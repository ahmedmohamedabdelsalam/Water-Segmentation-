"""Training utilities and trainer class."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import time

from .loss_functions import get_loss_function
from ..utils.logger import ExperimentLogger


class Trainer:
    """
    Trainer class for water segmentation models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        experiment_dir: Path,
        experiment_name: str,
        gradient_clip_max_norm: float = 0.5,
        log_interval: int = 10,
        save_checkpoints: bool = True,
        checkpoint_interval: int = 5,
        use_tensorboard: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            experiment_dir: Directory for saving experiments
            experiment_name: Name of experiment
            gradient_clip_max_norm: Max gradient norm for clipping
            log_interval: Log every N batches
            save_checkpoints: Whether to save checkpoints
            checkpoint_interval: Save checkpoint every N epochs
            use_tensorboard: Whether to use tensorboard
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.log_interval = log_interval
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        
        # Setup experiment directory
        self.experiment_dir = experiment_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = ExperimentLogger(experiment_dir, experiment_name)
        
        # Setup tensorboard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(self.experiment_dir / "tensorboard")
        
        # Track best model
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.best_epoch = 0
    
    def calculate_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Dice score.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Dice score
        """
        pred_binary = (pred > 0.5).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-7)
        
        return dice.item()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_max_norm
                )
            
            self.optimizer.step()
            
            # Calculate metrics
            batch_loss = loss.item()
            batch_dice = self.calculate_dice_score(outputs, masks)
            
            total_loss += batch_loss
            total_dice += batch_dice
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'dice': f'{batch_dice:.4f}'
            })
            
            # Log to tensorboard
            if self.use_tensorboard and (batch_idx + 1) % self.log_interval == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/BatchLoss', batch_loss, global_step)
                self.writer.add_scalar('Train/BatchDice', batch_dice, global_step)
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {
            'loss': avg_loss,
            'dice': avg_dice
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                batch_loss = loss.item()
                batch_dice = self.calculate_dice_score(outputs, masks)
                
                total_loss += batch_loss
                total_dice += batch_dice
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'dice': f'{batch_dice:.4f}'
                })
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return {
            'loss': avg_loss,
            'dice': avg_dice
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        if not self.save_checkpoints:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice
        }
        
        # Save regular checkpoint
        if epoch % self.checkpoint_interval == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.logger.info(f"Saved best model to {best_path}")
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 0.001,
        early_stopping_metric: str = "val_dice"
    ):
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Minimum improvement for early stopping
            early_stopping_metric: Metric to monitor ('val_dice' or 'val_loss')
        """
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            metrics_dict = {
                'train_loss': train_metrics['loss'],
                'train_dice': train_metrics['dice'],
                'val_loss': val_metrics['loss'],
                'val_dice': val_metrics['dice'],
                'lr': current_lr
            }
            self.logger.log_epoch(epoch, metrics_dict)
            
            # Log to tensorboard
            if self.use_tensorboard:
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Dice', train_metrics['dice'], epoch)
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Dice', val_metrics['dice'], epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Check for best model
            is_best = False
            if early_stopping_metric == "val_dice":
                if val_metrics['dice'] > self.best_val_dice + early_stopping_min_delta:
                    self.best_val_dice = val_metrics['dice']
                    self.best_epoch = epoch
                    is_best = True
                    patience_counter = 0
                    self.logger.log_best_model(epoch, val_metrics['dice'], 'Dice Score')
                else:
                    patience_counter += 1
            else:  # val_loss
                if val_metrics['loss'] < self.best_val_loss - early_stopping_min_delta:
                    self.best_val_loss = val_metrics['loss']
                    self.best_epoch = epoch
                    is_best = True
                    patience_counter = 0
                    self.logger.log_best_model(epoch, val_metrics['loss'], 'Val Loss')
                else:
                    patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                reason = f"No improvement in {early_stopping_metric} for {early_stopping_patience} epochs"
                self.logger.log_early_stopping(epoch, reason)
                break
        
        # Close tensorboard writer
        if self.use_tensorboard:
            self.writer.close()
        
        self.logger.logger.info(f"Training completed. Best epoch: {self.best_epoch}")
