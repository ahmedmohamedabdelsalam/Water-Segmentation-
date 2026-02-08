"""Configuration classes for training."""

from dataclasses import dataclass, field
from typing import Dict, Any
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    images_dir: str = "data/data/images"
    labels_dir: str = "data/data/labels"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class ModelConfig:
    """Model configuration."""
    type: str = "enhanced_unet"
    in_channels: int = 12
    out_channels: int = 1
    init_features: int = 64
    dropout_rate: float = 0.3


@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 10
    min_lr: float = 1e-5


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    patience: int = 15
    min_delta: float = 0.001
    metric: str = "val_dice"


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip_max_norm: float = 0.5
    loss_type: str = "combined"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"bce": 0.5, "dice": 0.5})
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    train: Dict[str, Any] = field(default_factory=lambda: {
        "enable": True,
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "rotation_90": 0.5
    })
    val: Dict[str, Any] = field(default_factory=lambda: {"enable": False})
    test: Dict[str, Any] = field(default_factory=lambda: {"enable": False})


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_interval: int = 10
    save_dir: str = "experiments"
    tensorboard: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 5


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            Config instance
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configs
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        
        # Training config with nested objects
        training_dict = config_dict.get('training', {})
        lr_scheduler_config = LRSchedulerConfig(**training_dict.get('lr_scheduler', {}))
        early_stopping_config = EarlyStoppingConfig(**training_dict.get('early_stopping', {}))
        training_config = TrainingConfig(
            **{k: v for k, v in training_dict.items() 
               if k not in ['lr_scheduler', 'early_stopping']},
            lr_scheduler=lr_scheduler_config,
            early_stopping=early_stopping_config
        )
        
        augmentation_config = AugmentationConfig(**config_dict.get('augmentation', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            augmentation=augmentation_config,
            logging=logging_config,
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'auto'),
            num_workers=config_dict.get('num_workers', 4),
            pin_memory=config_dict.get('pin_memory', True)
        )
