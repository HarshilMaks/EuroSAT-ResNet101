# Import the Necessary Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model import get_resnet50

# Setup the device for GPU usage if active
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Processed Data
train_data = torch.load("data/processed/train.pt")
val_data = torch.load("data/processed/val.pt")

# Wrap the dataset  
batch_size=32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


# Model Initialization
model: nn.Module = get_resnet50(num_classes=10, pretrained=True, freeze_backbone=True)
model = model.to(device)


# Loss and Optimizer functions
getloss = nn.CrossEntropyLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Fix: Remove torch.no_grad from accuracy function
def calculate_accuracy(predictions: torch.Tensor, true_labels: torch.Tensor) -> float:
    predicted_classes = predictions.argmax(dim=1)
    correct = (predicted_classes == true_labels).sum().item()
    total = true_labels.size(0)
    return correct / total if total > 0 else 0.0

# Training Loop
def train_batch(x: torch.Tensor, y: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, getloss: nn.Module):
    model.train()
    optimizer.zero_grad()
    prediction = model(x)
    batch_train_loss = getloss(prediction, y)
    batch_train_loss.backward()
    optimizer.step()
    acc = calculate_accuracy(prediction, y)
    return batch_train_loss.item(), acc


# Validation Loop
def validate_batch(x: torch.Tensor, y: torch.Tensor, model: nn.Module, getloss: nn.Module):
    model.eval()
    with torch.no_grad():
        evaluation = model(x)
        batch_eval_loss = getloss(evaluation, y)
        acc = calculate_accuracy(evaluation, y)
    return batch_eval_loss.item(), acc


def train(num_epochs: int = 10, save_path: str = "artifacts/best_model.pth"):
    best_val_acc = 0.0
    os.makedirs("artifacts", exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        # Train
        running_loss = 0.0
        running_acc = 0.0
        total_train = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            loss, acc = train_batch(images, labels, model, optimizer, getloss)
            batch_size_now = labels.size(0)
            running_loss += loss * batch_size_now
            running_acc += acc * batch_size_now
            total_train += batch_size_now

        train_loss = running_loss / max(total_train, 1)
        train_acc = running_acc / max(total_train, 1)

        # Validate
        val_running_loss = 0.0
        val_running_acc = 0.0
        total_val = 0
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            loss, acc = validate_batch(images, labels, model, getloss)
            batch_size_now = labels.size(0)
            val_running_loss += loss * batch_size_now
            val_running_acc += acc * batch_size_now
            total_val += batch_size_now

        val_loss = val_running_loss / max(total_val, 1)
        val_acc = val_running_acc / max(total_val, 1)

        scheduler.step()

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, save_path)
            print(f"Saved new best model with val_acc={val_acc:.4f} to {save_path}")


if __name__ == "__main__":
    train()
 