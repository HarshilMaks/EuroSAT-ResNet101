import argparse
import os
import sys
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torchvision.transforms as transforms  # type: ignore[import-not-found]

# project imports (assumes model.py exposes get_resnet101, get_model_info)
from model import get_resnet101, get_model_info

# Optional imports
try:
    import wandb  # type: ignore
    _wandb_available = True
except Exception:
    wandb = None  # type: ignore
    _wandb_available = False

try:
    # torchvision weights utilities (used for loading pretrained backbone)
    import torchvision.models as tv_models  # type: ignore
    from torchvision.models import ResNet101_Weights  # type: ignore
    _torchvision_available = True
except Exception:
    tv_models = None  # type: ignore
    ResNet101_Weights = None  # type: ignore
    _torchvision_available = False

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Progressive training for custom ResNet-101 (EuroSAT)")
    # runtime toggles
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging (writes runs/)")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (requires wandb installed)")
    p.add_argument("--wandb-project", default="eurosat-resnet", help="W&B project name")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (user or team)")
    p.add_argument("--pretrained-backbone", choices=["imagenet", "none"], default="none",
                   help="If 'imagenet', load torchvision ResNet101 ImageNet backbone weights into custom model (best-effort).")
    # data / training
    p.add_argument("--data-dir", default="data/processed", help="Path to processed data (train.pt, val.pt)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    # phase-specific overrides
    p.add_argument("--phase1-epochs", type=int, default=15)
    p.add_argument("--phase2-epochs", type=int, default=20)
    p.add_argument("--phase3-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-3, help="Base LR for classifier (backbone will use 0.1 * LR when unfrozen)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Preprocessing / Dataset wrappers
# ---------------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels.long()
        self.transform = transform
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            # if saved tensors were normalized, unnormalize first
            image = image * self._std + self._mean
            image = torch.clamp(image, 0.0, 1.0)
            image = self.transform(image)
        return image, label

# label smoothing loss as function
def label_smoothing_cross_entropy(x: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    confidence = 1.0 - smoothing
    logprobs = torch.log_softmax(x, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()

def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    for name, param in model.named_parameters():
        if any(ln in name for ln in layer_names):
            param.requires_grad = True
            print(f"Unfroze: {name}")

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------
# Pretrained backbone loader (best-effort)
# ---------------------------
def load_torchvision_backbone_weights_into_custom(model: nn.Module, weights_choice: str = "imagenet") -> Dict[str, int]:
    """
    Best-effort copy of torchvision ResNet101 backbone weights into a custom ResNet101 model.
    Returns a small report dict with counts: copied, mismatched, skipped.
    Notes:
      - This function copies conv1, bn1, layer1..layer4 params where names and shapes match.
      - It intentionally skips classifier/fc mapping because custom classifier often differs.
      - Requires torchvision to be importable.
    """
    report = {"copied": 0, "shape_mismatch": 0, "skipped": 0}

    if not _torchvision_available:
        print("torchvision not available — cannot load pretrained backbone.")
        report["skipped"] = 1
        return report

    if weights_choice != "imagenet":
        print(f"Unknown weights_choice={weights_choice} — skipping.")
        report["skipped"] = 1
        return report

    print("Loading torchvision ResNet101 (ImageNet weights) for mapping...")
    tv_weights = ResNet101_Weights.DEFAULT
    tv_model = tv_models.resnet101(weights=tv_weights)
    tv_state = tv_model.state_dict()
    custom_state = model.state_dict()

    # Candidate prefixes to copy (backbone only)
    prefixes = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    copied_keys = []
    mismatched_keys = []

    for k, v in tv_state.items():
        if any(k.startswith(pref) for pref in prefixes):
            if k in custom_state and custom_state[k].shape == v.shape:
                custom_state[k].copy_(v)
                report["copied"] += 1
                copied_keys.append(k)
            else:
                # try alternate: sometimes custom implementation uses different internal naming
                report["shape_mismatch"] += 1
                mismatched_keys.append(k)

    # Load the modified state back into model (this will only change tensors we copied)
    model.load_state_dict(custom_state, strict=False)
    print(f"Backbone mapping done. Copied: {report['copied']}, shape mismatches/skipped: {report['shape_mismatch']}")
    if report["shape_mismatch"] > 0:
        print("Note: Some layers couldn't be copied due to naming/shape mismatch — classifier/head remains untouched.")
    return report

# ---------------------------
# train_phase (single reusable function)
# ---------------------------
def train_phase(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    phase_name: str,
    device: torch.device,
    save_path: str, #
    use_amp: bool = True,
    warmup_epochs: int = 3,
    tb_writer: Optional[SummaryWriter] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"STARTING {phase_name.upper()}")
    print(f"{'='*60}")
    print(f"Trainable parameters: {count_trainable_params(model):,}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")

    criterion_fn = lambda x, y: label_smoothing_cross_entropy(x, y, smoothing=0.1)

    # parameter groups
    classifier_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name or name.startswith("classifier"):
                classifier_params.append(param)
            else:
                backbone_params.append(param)

    if backbone_params:
        optimizer = optim.AdamW([
            {"params": classifier_params, "lr": lr},
            {"params": backbone_params, "lr": lr * 0.1}
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(classifier_params, lr=lr, weight_decay=1e-4)

    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_epochs))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=lr * 0.01)

    best_val_acc = 0.0
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    for epoch in range(1, num_epochs + 1):
        print(f"\n{phase_name} - Epoch {epoch}/{num_epochs}")

        # --- train ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_dataloader, desc="Training")
        for images, labels in train_pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(images)
                    loss = criterion_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size_now = labels.size(0)
            running_loss += float(loss.item()) * batch_size_now
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += batch_size_now
            current_acc = correct_train / total_train if total_train > 0 else 0.0
            train_pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{current_acc:.4f}')

        train_loss = running_loss / max(1, total_train)
        train_acc = correct_train / max(1, total_train)

        # --- validate ---
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc="Validation")
            for images, labels in val_pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                if scaler is not None:
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = model(images)
                        loss = criterion_fn(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion_fn(outputs, labels)

                batch_size_now = labels.size(0)
                val_running_loss += float(loss.item()) * batch_size_now
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += batch_size_now
                current_acc = correct_val / total_val if total_val > 0 else 0.0
                val_pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{current_acc:.4f}')

        val_loss = val_running_loss / max(1, total_val)
        val_acc = correct_val / max(1, total_val)

        # scheduler step
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # logging
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        print(f"           Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        print(f"           LR: {current_lr:.6f}")

        # TensorBoard logging
        if tb_writer is not None:
            tb_writer.add_scalar(f"{phase_name}/train_loss", train_loss, epoch)
            tb_writer.add_scalar(f"{phase_name}/val_loss", val_loss, epoch)
            tb_writer.add_scalar(f"{phase_name}/train_acc", train_acc, epoch)
            tb_writer.add_scalar(f"{phase_name}/val_acc", val_acc, epoch)
            tb_writer.add_scalar(f"{phase_name}/lr", current_lr, epoch)

        # Weights & Biases logging
        if wandb_run is not None:
            wandb_run.log({
                f"{phase_name}/train_loss": train_loss,
                f"{phase_name}/val_loss": val_loss,
                f"{phase_name}/train_acc": train_acc,
                f"{phase_name}/val_acc": val_acc,
                "lr": current_lr,
                "epoch": epoch
            })

        # checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "phase": phase_name,
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
            print(f"Saved new best model: {save_path} (val_acc={val_acc:.4f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }

# ---------------------------
# Main progressive training
# ---------------------------
def main() -> Dict[str, Dict[str, Any]]:
    args = parse_args()
    set_seed(args.seed)
    device = safe_device()
    print(f"Using device: {device}")
    os.makedirs("artifacts", exist_ok=True)

    # load data
    train_path = os.path.join(args.data_dir, "train.pt")
    val_path = os.path.join(args.data_dir, "val.pt")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Processed data not found at {args.data_dir}. Expected train.pt and val.pt.")
        sys.exit(1)

    print("Loading data...")
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    print(f"Train images: {train_data['images'].shape}, labels: {train_data['labels'].shape}")
    print(f"Val images:   {val_data['images'].shape}, labels: {val_data['labels'].shape}")

    # dataloaders
    train_dataset = AugmentedDataset(train_data["images"], train_data["labels"], transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = TensorDataset(val_data["images"], val_data["labels"].long())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers//2), pin_memory=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Experiment logging setup
    tb_writer = SummaryWriter() if args.tensorboard else None
    wandb_run = None
    if args.wandb:
        if not _wandb_available:
            print("W&B requested but `wandb` package is not installed. Install with `pip install wandb` or omit --wandb.")
            args.wandb = False
        else:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config={
                "batch_size": args.batch_size,
                "phase1_epochs": args.phase1_epochs,
                "phase2_epochs": args.phase2_epochs,
                "phase3_epochs": args.phase3_epochs,
                "base_lr": args.lr,
            })
            wandb_run = wandb

    # initialize model (custom ResNet-101)
    print("Initializing ResNet-101 model (custom implementation) with frozen backbone...")
    model = get_resnet101(
        num_classes=10,
        dropout_rate=0.3,
        hidden_size=1024,
        freeze_backbone=True
    )
    model = model.to(device)
    get_model_info(model)

    # optionally load torchvision backbone
    if args.pretrained_backbone == "imagenet":
        print("Attempting to load torchvision ImageNet backbone weights into custom model (best-effort)...")
        report = load_torchvision_backbone_weights_into_custom(model, weights_choice="imagenet")
        if wandb_run is not None:
            wandb_run.config.update({"pretrained_backbone_report": report})

    results: Dict[str, Dict[str, Any]] = {}

    # PHASE 1
    phase1_save = "artifacts/phase1_model.pth"
    print("\nPHASE 1: Training classifier only...")
    res1 = train_phase(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=args.phase1_epochs,
        lr=args.lr,
        phase_name="Phase 1 - Classifier Only",
        device=device,
        save_path=phase1_save,
        use_amp=True,
        warmup_epochs=3,
        tb_writer=tb_writer,
        wandb_run=wandb_run
    )
    results["phase1"] = res1

    # PHASE 2: unfreeze last residual stage + classifier
    print("\nPHASE 2: Unfreezing last layer...")
    unfreeze_layers(model, ["layer4", "classifier"])
    get_model_info(model)
    phase2_save = "artifacts/phase2_model.pth"
    res2 = train_phase(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=args.phase2_epochs,
        lr=args.lr * 0.33,  # reduce base LR for safety in fine-tuning
        phase_name="Phase 2 - Last Layer + Classifier",
        device=device,
        save_path=phase2_save,
        use_amp=True,
        warmup_epochs=3,
        tb_writer=tb_writer,
        wandb_run=wandb_run
    )
    results["phase2"] = res2

    # PHASE 3: full fine-tune
    print("\nPHASE 3: Full fine-tuning...")
    unfreeze_layers(model, ["layer1", "layer2", "layer3", "conv1", "bn1", "classifier"])
    get_model_info(model)
    phase3_save = "artifacts/final_model.pth"
    res3 = train_phase(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=args.phase3_epochs,
        lr=args.lr * 0.1,
        phase_name="Phase 3 - Full Fine-tuning",
        device=device,
        save_path=phase3_save,
        use_amp=True,
        warmup_epochs=2,
        tb_writer=tb_writer,
        wandb_run=wandb_run
    )
    results["phase3"] = res3

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Phase 1 Best Accuracy: {res1['best_val_acc']:.1%}")
    print(f"Phase 2 Best Accuracy: {res2['best_val_acc']:.1%}")
    print(f"Phase 3 Best Accuracy: {res3['best_val_acc']:.1%}")

    if res3["best_val_acc"] >= 0.85:
        print("SUCCESS: Achieved target accuracy!")
    else:
        print("Consider additional training, augmentations or learning-rate tuning.")

    # close writers / wandb
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    return results

if __name__ == "__main__":
    _ = main()