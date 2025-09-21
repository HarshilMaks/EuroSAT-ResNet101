# Import the Necessary Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from model import get_resnet101, get_model_info  # type: ignore
from typing import Any, Optional, Dict, List, Union

# Setup the device for GPU usage if active
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Processed Data
print("Loading data...")
train_data = torch.load("data/processed/train.pt")
val_data = torch.load("data/processed/val.pt")
print(f"Train data shape: {train_data['images'].shape}, Labels: {train_data['labels'].shape}")
print(f"Val data shape: {val_data['images'].shape}, Labels: {val_data['labels'].shape}")

# Wrap the dataset  
batch_size = 32
images, labels = train_data["images"], train_data["labels"]
train_dataset = TensorDataset(images, labels.long())
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)

val_dataset = TensorDataset(val_data["images"], val_data["labels"].long())
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True
)

print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

# Model Initialization - FIXED: Removed pretrained=True parameter
print("Initializing ResNet-101 model...")
model: nn.Module = get_resnet101(
    num_classes=10, 
    dropout_rate=0.2,
    hidden_size=512,
    freeze_backbone=True
)
model = model.to(device)
print(f"Model loaded on {device}")

# Get model info using our function
get_model_info(model)

# Loss and Optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-3,
    weight_decay=1e-4  # Added weight decay for regularization
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

print(f"Optimizer learning rate: {optimizer.param_groups[0]['lr']}")

# Accuracy Function
def calculate_accuracy(predictions: torch.Tensor, true_labels: torch.Tensor) -> float:
    """Calculate accuracy from predictions and true labels."""
    predicted_classes = predictions.argmax(dim=1)
    correct = (predicted_classes == true_labels).sum().item()
    total = true_labels.size(0)
    return correct / total if total > 0 else 0.0

# Training step function
def train_batch(
    x: torch.Tensor, 
    y: torch.Tensor, 
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device, 
    scaler: Optional[Any] = None
):
    """Execute one training batch."""
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long()
    
    optimizer.zero_grad()
    
    if scaler is not None:
        # Use Automatic Mixed Precision when scaler is provided
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
    
    # Move predictions and labels to CPU for accuracy calculation
    preds_cpu = outputs.argmax(dim=1).detach().cpu()
    labels_cpu = y.detach().cpu()
    
    return loss.item(), preds_cpu, labels_cpu

# Validation step function
def validate_batch(
    x: torch.Tensor, 
    y: torch.Tensor, 
    model: nn.Module, 
    criterion: nn.Module, 
    device: torch.device
):
    """Execute one validation batch."""
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long()
    
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Move predictions and labels to CPU
    preds_cpu = outputs.argmax(dim=1).detach().cpu()
    labels_cpu = y.detach().cpu()
    
    return loss.item(), preds_cpu, labels_cpu

def train(
    num_epochs: int = 20, 
    save_path: str = "artifacts/best_model.pth", 
    patience: int = 5,
    use_amp: bool = False
) -> Dict[str, Union[float, List[float]]]:
    """Main training loop with improvements."""
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Early stopping patience: {patience}")
    print(f"Using AMP: {use_amp}")
    
    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)
    
    # Initialize variables
    best_val_acc: float = 0.0
    patience_counter: int = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    
    # Initialize AMP scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None  # type: ignore
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_dataloader, desc=f"Training")
        for images, labels in train_pbar:
            loss, preds_cpu, labels_cpu = train_batch(
                images, labels, model, optimizer, criterion, device, scaler
            )
            
            batch_size_now = labels_cpu.size(0)
            running_loss += loss * batch_size_now
            correct_train += (preds_cpu == labels_cpu).sum().item()
            total_train += batch_size_now
            
            # Update progress bar
            current_acc = correct_train / total_train
            train_pbar.set_postfix(loss=f'{loss:.4f}', acc=f'{current_acc:.4f}')  # type: ignore

        train_loss = running_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(val_dataloader, desc=f"Validation")
        with torch.no_grad():
            for images, labels in val_pbar:
                loss, preds_cpu, labels_cpu = validate_batch(
                    images, labels, model, criterion, device
                )
                
                batch_size_now = labels_cpu.size(0)
                val_running_loss += loss * batch_size_now
                correct_val += (preds_cpu == labels_cpu).sum().item()
                total_val += batch_size_now
                
                # Update progress bar
                current_acc = correct_val / total_val
                val_pbar.set_postfix(loss=f'{loss:.4f}', acc=f'{current_acc:.4f}')  # type: ignore

        val_loss = val_running_loss / max(total_val, 1)
        val_acc = correct_val / max(total_val, 1)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch:02d}/{num_epochs} Summary:")
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"LR: {current_lr:.6f}")
            
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model checkpoint
            checkpoint: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            }
            
            torch.save(checkpoint, save_path)
            print(f"✓ Saved new best model (val_acc={val_acc:.4f}) to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
        
        # Clean up GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy achieved: {best_val_acc:.4f}")
    return {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

if __name__ == "__main__":
    # Run training
    results: Dict[str, Union[float, List[float]]] = train(
        num_epochs=20,
        patience=7,
        use_amp=False  # Set to True if you want to use Automatic Mixed Precision
    )