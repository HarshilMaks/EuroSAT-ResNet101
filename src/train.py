# Import the Necessary Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model import get_resnet50
from typing import Any, Optional

# Setup the device for GPU usage if active
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Processed Data
train_data = torch.load("data/processed/train.pt")
val_data = torch.load("data/processed/val.pt")

# Wrap the dataset  
batch_size=32
from torch.utils.data import TensorDataset
images, labels = train_data["images"], train_data["labels"]
train_dataset = TensorDataset(images, labels.long())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataset = TensorDataset(val_data["images"], val_data["labels"].long())
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Model Initialization
model: nn.Module = get_resnet50(num_classes=10, pretrained=True, freeze_backbone=True)
model = model.to(device)
print(f"Model loaded on {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# Loss and Optimizer functions
getloss = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# Accuracy Function
def calculate_accuracy(predictions: torch.Tensor, true_labels: torch.Tensor) -> float:
    predicted_classes = predictions.argmax(dim=1)
    correct = (predicted_classes == true_labels).sum().item()
    total = true_labels.size(0)
    return correct / total if total > 0 else 0.0

#Tqdm import
from tqdm import tqdm

# Training step function
def train_batch(x: torch.Tensor, y: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, scaler: Optional[Any] = None):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long()
    optimizer.zero_grad()
    if scaler is not None:
        # Use AMP when a scaler is provided
        with torch.autocast(device_type="cuda", enabled=True):
            outputs = model(x)
            loss = criterion(outputs, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    preds_cpu = outputs.argmax(dim=1).detach().cpu()
    labels_cpu = y.detach().cpu()
    return loss.item(), preds_cpu, labels_cpu


# Validation step function
def validate_batch(x: torch.Tensor, y: torch.Tensor, model: nn.Module, criterion: nn.Module, device: torch.device):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long()
    outputs = model(x)
    loss = criterion(outputs, y)
    preds_cpu = outputs.argmax(dim=1).detach().cpu()
    labels_cpu = y.detach().cpu()
    return loss.item(), preds_cpu, labels_cpu


def train(num_epochs: int = 20, save_path: str = "artifacts/best_model.pth", patience: int = 5):
    best_val_acc = 0.0
    patience_counter = 0
    os.makedirs("artifacts", exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        model.train()
        # Train
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch} Training"):
            loss, preds_cpu, labels_cpu = train_batch(images, labels, model, optimizer, getloss, device, scaler=None)
            batch_size_now = labels_cpu.size(0)
            running_loss += loss * batch_size_now
            correct_train += (preds_cpu == labels_cpu).sum().item()
            total_train += batch_size_now

        train_loss = running_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        # Validate
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch} Validation"):
                loss, preds_cpu, labels_cpu = validate_batch(images, labels, model, getloss, device)
                batch_size_now = labels_cpu.size(0)
                val_running_loss += loss * batch_size_now
                correct_val += (preds_cpu == labels_cpu).sum().item()
                total_val += batch_size_now

        val_loss = val_running_loss / max(total_val, 1)
        val_acc = correct_val / max(total_val, 1)

        scheduler.step()

        print(f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 # Reset patience
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, save_path)
            print(f"Saved new best model with val_acc={val_acc:.4f} to {save_path}")
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
if __name__ == "__main__":
    train()