import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.model import get_resnet101  # your model.py function
from datasets import load_from_disk
from PIL import Image

# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Helpers for Dataset
# =========================
class EuroSATDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loader(data_dir, batch_size=32):
    dataset = load_from_disk(data_dir)["train"]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    euro_dataset = EuroSATDataset(dataset, transform=transform)
    loader = DataLoader(euro_dataset, batch_size=batch_size, shuffle=False)
    return euro_dataset, loader

# =========================
# Visualization Functions
# =========================
def visualize_dataset_samples(dataset, save_path="assets/eurosat_rgb_preview.png", n=16):
    os.makedirs("assets", exist_ok=True)
    indices = random.sample(range(len(dataset)), n)
    images, labels = zip(*[dataset[i] for i in indices])

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, img, label in zip(axes.flatten(), images, labels):
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                      np.array([0.485, 0.456, 0.406]), 0, 1)
        ax.imshow(img)
        ax.set_title(str(label), fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved dataset preview to {save_path}")

def visualize_predictions(model, dataset, save_path="assets/eurosat_rgb_predictions.png", n=9):
    os.makedirs("assets", exist_ok=True)
    indices = random.sample(range(len(dataset)), n)
    images, labels = zip(*[dataset[i] for i in indices])

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, img, label in zip(axes.flatten(), images, labels):
        input_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) +
                         np.array([0.485, 0.456, 0.406]), 0, 1)
        ax.imshow(img_np)
        ax.set_title(f"GT: {label}\nPred: {pred.item()}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved sample predictions to {save_path}")

def plot_confusion_matrix(model, loader, save_path="assets/eurosat_rgb_confusion_matrix.png"):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

# =========================
# Load Model
# =========================
def load_model(model_path, num_classes):
    model = get_resnet101(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# =========================
# Main
# =========================
def main():
    data_dir = "data/raw/eurosat_rgb"
    model_path = "artifacts/best_model.pth"
    num_classes = 10

    dataset, loader = get_data_loader(data_dir)
    visualize_dataset_samples(dataset)
    model = load_model(model_path, num_classes)
    visualize_predictions(model, dataset)
    plot_confusion_matrix(model, loader)

if __name__ == "__main__":
    main()
