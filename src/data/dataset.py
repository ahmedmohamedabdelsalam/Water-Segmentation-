"""Dataset class for water segmentation from satellite imagery."""

import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import cv2
import rasterio
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import AdvancedImageTransform, normalize_multispectral_image, ToTensor


class WaterSegmentationDataset(Dataset):
    """
    Dataset for water segmentation from multispectral satellite imagery.
    
    Attributes:
        images_dir: Path to directory containing .tif images
        labels_dir: Path to directory containing .png masks
        matched_files: List of (image_file, label_file) tuples
        transform: Augmentation transform
        to_tensor: Tensor conversion transform
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        file_list: Optional[List[str]] = None,
        transform: Optional[AdvancedImageTransform] = None,
        target_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing .tif images
            labels_dir: Directory containing .png label masks
            file_list: Optional list of file basenames to include
            transform: Augmentation transform
            target_size: Target (height, width) for resizing
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.target_size = target_size
        self.to_tensor = ToTensor()
        
        # Match images and labels
        if file_list is not None:
            self.matched_files = [(f"{name}.tif", f"{name}.png") for name in file_list]
        else:
            self.matched_files = self._match_images_and_labels()
    
    def _match_images_and_labels(self) -> List[Tuple[str, str]]:
        """
        Match image files with their corresponding label files.
        
        Returns:
            List of (image_filename, label_filename) tuples
        """
        # Get all files
        image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])
        label_files = sorted([f for f in os.listdir(self.labels_dir) if f.endswith('.png')])
        
        # Extract basenames
        image_names = {os.path.splitext(f)[0] for f in image_files}
        label_names = {os.path.splitext(f)[0] for f in label_files}
        
        # Find common names
        common_names = image_names.intersection(label_names)
        
        matched_files = [(f"{name}.tif", f"{name}.png") for name in sorted(common_names)]
        
        print(f"Matched {len(matched_files)} image-label pairs")
        
        return matched_files
    
    def _load_multispectral_image(self, image_path: Path) -> np.ndarray:
        """
        Load multispectral .tif image.
        
        Args:
            image_path: Path to .tif file
            
        Returns:
            Normalized image array (H, W, C)
        """
        try:
            with rasterio.open(image_path) as src:
                # Read all bands
                image = src.read()  # Shape: (C, H, W)
                # Transpose to (H, W, C)
                image = image.transpose(1, 2, 0)
            
            # Resize
            image = cv2.resize(image, self.target_size)
            
            # Normalize
            image = normalize_multispectral_image(image)
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return zero array to avoid breaking the data loader
            # In production, you might want to handle this differently
            return np.zeros((*self.target_size, 12), dtype=np.float32)
    
    def _load_label(self, label_path: Path) -> np.ndarray:
        """
        Load binary label mask.
        
        Args:
            label_path: Path to .png label file
            
        Returns:
            Binary mask array (H, W)
        """
        try:
            label = Image.open(label_path).convert('L')
            label = np.array(label)
            
            # Resize
            label = cv2.resize(label, self.target_size)
            
            # Binarize (anything > 0 is water)
            label = (label > 0).astype(np.float32)
            
            return label
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            return np.zeros(self.target_size, dtype=np.float32)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.matched_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        image_file, label_file = self.matched_files[idx]
        
        # Load image and label
        image = self._load_multispectral_image(self.images_dir / image_file)
        label = self._load_label(self.labels_dir / label_file)
        
        # Apply augmentation if specified
        if self.transform:
            image, label = self.transform(image, label)
        
        # Convert to tensor
        image_tensor, label_tensor = self.to_tensor(image, label)
        
        return image_tensor, label_tensor


def create_dataloaders(
    images_dir: str,
    labels_dir: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    augmentation_prob: float = 0.5
) -> Tuple:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducibility
        augmentation_prob: Probability for augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create full dataset to get file list
    temp_dataset = WaterSegmentationDataset(images_dir, labels_dir)
    all_files = [os.path.splitext(img)[0] for img, _ in temp_dataset.matched_files]
    
    # Split files
    np.random.seed(seed)
    np.random.shuffle(all_files)
    
    n_total = len(all_files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    print(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Create datasets with appropriate transforms
    train_dataset = WaterSegmentationDataset(
        images_dir, labels_dir,
        file_list=train_files,
        transform=AdvancedImageTransform(p=augmentation_prob, train=True)
    )
    
    val_dataset = WaterSegmentationDataset(
        images_dir, labels_dir,
        file_list=val_files,
        transform=AdvancedImageTransform(p=0.0, train=False)
    )
    
    test_dataset = WaterSegmentationDataset(
        images_dir, labels_dir,
        file_list=test_files,
        transform=AdvancedImageTransform(p=0.0, train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
