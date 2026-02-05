"""
Utility functions and constants for the EuroSAT-ResNet101 project.

This module centralizes common functionalities such as setting random seeds,
configuring device placement (CPU/GPU), and handling data loading and image
manipulation to avoid code duplication across different scripts.
"""

import os
import random
import torch
import numpy as np
from typing import Dict, Optional, Tuple

# =================================================================================================
# Project-wide Constants
# =================================================================================================

# EuroSAT class names for consistent labeling in visualizations and outputs.
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Standard ImageNet normalization statistics used for preprocessing and denormalizing images.
IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}


# =================================================================================================
# Environment and Setup Utilities
# =================================================================================================

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across PyTorch, NumPy, and Python's random module.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # The following two lines are often used for reproducibility, but can impact performance.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to {seed}")


def safe_device() -> torch.device:
    """
    Checks for CUDA availability and returns the appropriate torch.device.

    Returns:
        torch.device: The torch.device object ('cuda' if available, otherwise 'cpu').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# =================================================================================================
# Data Handling Utilities
# =================================================================================================

def load_processed_data(data_dir: str = "data/processed") -> Optional[Tuple[Dict, Dict]]:
    """
    Loads the processed training and validation tensor data from .pt files.

    Args:
        data_dir (str): The directory where 'train.pt' and 'val.pt' are stored.

    Returns:
        A tuple containing the training and validation data dictionaries,
        or (None, None) if the files are not found.
    """
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Error: Processed data not found in '{data_dir}'.")
        print("Please run the preprocessing script first.")
        return None, None

    try:
        train_data = torch.load(train_path)
        val_data = torch.load(val_path)
        print(f"Successfully loaded processed data from '{data_dir}'")
        return train_data, val_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


# =================================================================================================
# Visualization Utilities
# =================================================================================================

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalizes an image tensor using ImageNet stats for visualization.

    Args:
        tensor (torch.Tensor): The input image tensor (C, H, W) with ImageNet normalization.

    Returns:
        np.ndarray: The denormalized image as a NumPy array (H, W, C) ready for plotting.
    """
    mean = torch.tensor(IMAGENET_STATS["mean"]).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STATS["std"]).view(3, 1, 1)

    # Move tensors to the same device to avoid errors
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)

    # Denormalize, clamp to valid range [0, 1], and convert for plotting
    denorm_tensor = tensor * std + mean
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)

    # Convert to NumPy array and change dimension order from (C, H, W) to (H, W, C)
    return denorm_tensor.cpu().permute(1, 2, 0).numpy()
