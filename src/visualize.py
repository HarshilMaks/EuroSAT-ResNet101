import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.model import get_resnet101, get_model_info
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# EuroSAT class names for better visualization
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

def load_processed_data(data_dir="data/processed"):
    """Load processed tensor data from .pt files."""
    try:
        train_data = torch.load(os.path.join(data_dir, "train.pt"))
        val_data = torch.load(os.path.join(data_dir, "val.pt"))
        
        print(f"Train data shape: {train_data['images'].shape}")
        print(f"Val data shape: {val_data['images'].shape}")
        
        return train_data, val_data
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        print("Make sure you have run the preprocessing script first.")
        return None, None

def get_data_loader(data, batch_size=32, shuffle=False):
    """Create DataLoader from processed tensor data."""
    images = data["images"]
    labels = data["labels"].long()
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader

def denormalize_image(tensor):
    """Denormalize image tensor for visualization."""
    # ImageNet normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose for matplotlib
    return tensor.permute(1, 2, 0).numpy()

def visualize_dataset_samples(dataset, save_path="assets/eurosat_rgb_preview.png", n=16):
    """Visualize random samples from the dataset."""
    os.makedirs("assets", exist_ok=True)
    
    # Randomly sample indices
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    
    # Create subplot grid
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Denormalize image
        img_np = denormalize_image(img)
        
        # Plot
        axes[i].imshow(img_np)
        axes[i].set_title(f"{CLASS_NAMES[label]}", fontsize=10)
        axes[i].axis("off")
    
    # Hide unused subplots
    for j in range(len(indices), len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dataset preview to {save_path}")

def visualize_predictions(model, dataset, save_path="assets/eurosat_rgb_predictions.png", n=9):
    """Visualize model predictions on random samples."""
    os.makedirs("assets", exist_ok=True)
    
    model.eval()
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, true_label = dataset[idx]
            
            # Get prediction
            input_tensor = img.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            _, pred_label = torch.max(outputs, 1)
            pred_label = pred_label.item()
            
            # Get confidence (softmax probability)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probs.max().item()
            
            # Denormalize image
            img_np = denormalize_image(img)
            
            # Plot
            axes[i].imshow(img_np)
            
            # Color code: green if correct, red if wrong
            color = 'green' if pred_label == true_label else 'red'
            axes[i].set_title(
                f"True: {CLASS_NAMES[true_label]}\n"
                f"Pred: {CLASS_NAMES[pred_label]}\n"
                f"Conf: {confidence:.2f}",
                fontsize=9, color=color
            )
            axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample predictions to {save_path}")

def plot_confusion_matrix(model, loader, save_path="assets/eurosat_rgb_confusion_matrix.png"):
    """Generate and plot confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Computing predictions for confusion matrix...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(loader)} batches")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=CLASS_NAMES
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix - ResNet-101 on EuroSAT", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")
    
    # Print accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Overall Accuracy: {accuracy:.1%}")
    
    return cm

def load_trained_model(model_path, num_classes=10):
    """Load trained ResNet-101 model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    # Create model architecture
    model = get_resnet101(
        num_classes=num_classes, 
        dropout_rate=0.2, 
        hidden_size=512,
        freeze_backbone=False
    )
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Best validation accuracy: {checkpoint.get('val_acc', 'unknown'):.1%}")
        else:
            state_dict = checkpoint
            print("Loaded state dict directly")
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Print model info
        get_model_info(model)
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first using train.py")
        return None

def compute_class_accuracy(cm, class_names):
    """Compute per-class accuracy from confusion matrix."""
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i, (name, acc) in enumerate(zip(class_names, class_acc)):
        print(f"{name:20s}: {acc:.1%}")

def main():
    """Main evaluation and visualization function."""
    print("Starting visualization and evaluation...")
    
    # Configuration
    data_dir = "data/processed"
    model_path = "artifacts/best_model.pth"
    num_classes = 10
    batch_size = 32
    
    # Load data
    print("Loading processed data...")
    train_data, val_data = load_processed_data(data_dir)
    if train_data is None or val_data is None:
        return
    
    # Create datasets and loaders
    train_dataset, train_loader = get_data_loader(train_data, batch_size=batch_size)
    val_dataset, val_loader = get_data_loader(val_data, batch_size=batch_size)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Generate dataset preview
    print("\nGenerating dataset preview...")
    visualize_dataset_samples(train_dataset, n=16)
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_trained_model(model_path, num_classes)
    if model is None:
        return
    
    # Generate predictions visualization
    print("\nGenerating prediction samples...")
    visualize_predictions(model, val_dataset, n=9)
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(model, val_loader)
    
    # Compute per-class accuracy
    compute_class_accuracy(cm, CLASS_NAMES)
    
    print("\nVisualization complete! Check the 'assets' folder for generated images.")

if __name__ == "__main__":
    main()