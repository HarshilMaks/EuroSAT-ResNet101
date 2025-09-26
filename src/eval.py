"""
Model Evaluation Script

This script loads a trained model checkpoint and evaluates its performance on the
validation dataset. It computes and prints key metrics including overall accuracy,
per-class precision, recall, F1-score, and a confusion matrix.

This provides a quantitative assessment of the model's performance, complementing
the visual analysis in `visualize.py`.
"""

import argparse
import sys
import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path to allow direct script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_resnet101, get_model_info
from src.utils import CLASS_NAMES, load_processed_data, safe_device

def parse_args():
    """Parse command-line arguments for the evaluation script."""
    p = argparse.ArgumentParser(description="Evaluate a trained ResNet-101 model on the EuroSAT dataset.")
    p.add_argument(
        "--model-path",
        type=str,
        default="artifacts/final_model.pth",
        help="Path to the trained model checkpoint (.pth file)."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing the processed validation data (val.pt)."
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation."
    )
    return p.parse_args()


def load_model_for_eval(model_path: str, num_classes: int, device: torch.device):
    """Loads a trained model from a checkpoint for evaluation."""
    print(f"Loading model from checkpoint: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at '{model_path}'")
        return None

    # Initialize the model architecture. Parameters like dropout don't matter for eval.
    model = get_resnet101(num_classes=num_classes, hidden_size=1024, freeze_backbone=False)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully.")
        get_model_info(model)
        return model
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return None


def evaluate(model, data_loader, device):
    """Runs the model on the dataset and collects predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    print("\nStarting evaluation...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i + 1) % 50 == 0:
                print(f"  Processed batch {i + 1}/{len(data_loader)}")

    print("Evaluation complete.")
    return np.array(all_labels), np.array(all_preds)


def main():
    """Main function to run the model evaluation."""
    args = parse_args()
    device = safe_device()

    # Load data
    _, val_data = load_processed_data(args.data_dir)
    if val_data is None:
        sys.exit(1)

    val_dataset = TensorDataset(val_data["images"], val_data["labels"].long())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(val_dataset)} validation samples.")

    # Load model
    model = load_model_for_eval(args.model_path, num_classes=len(CLASS_NAMES), device=device)
    if model is None:
        sys.exit(1)

    # Get predictions
    true_labels, predicted_labels = evaluate(model, val_loader, device)

    # --- Calculate and Display Metrics ---
    print("\n" + "="*50)
    print("              Model Performance Metrics")
    print("="*50)

    # Overall Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    # Classification Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        digits=3
    )
    print(report)

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)
    print("\n" + "="*50)


if __name__ == "__main__":
    main()
