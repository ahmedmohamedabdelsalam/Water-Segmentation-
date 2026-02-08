"""Logging utilities for experiment tracking."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None and log_file is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """
    Experiment logger for tracking metrics and progress.
    """
    
    def __init__(self, experiment_dir: Path, experiment_name: str):
        """
        Initialize experiment logger.
        
        Args:
            experiment_dir: Base directory for experiments
            experiment_name: Name of the experiment
        """
        self.experiment_dir = experiment_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.experiment_dir / "training.log"
        self.logger = setup_logger(
            f"experiment_{experiment_name}",
            self.experiment_dir,
            "training.log"
        )
        
    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""
        self.logger.info("="*80)
        self.logger.info(f"Experiment Configuration:")
        self.logger.info("="*80)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("="*80)
    
    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch metrics."""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:3d} | {metric_str}")
    
    def log_best_model(self, epoch: int, metric_value: float, metric_name: str) -> None:
        """Log best model information."""
        self.logger.info("="*80)
        self.logger.info(f"New best model at epoch {epoch}: {metric_name} = {metric_value:.4f}")
        self.logger.info("="*80)
    
    def log_early_stopping(self, epoch: int, reason: str) -> None:
        """Log early stopping."""
        self.logger.info("="*80)
        self.logger.info(f"Early stopping triggered at epoch {epoch}")
        self.logger.info(f"Reason: {reason}")
        self.logger.info("="*80)
