"""
CropGuard AI — Training Pipeline
MobileNetV2 transfer learning on PlantVillage dataset.

Usage:
    python training/train.py --data_dir data/plantvillage --epochs 20

Two-phase training:
    Phase 1 (backbone frozen): Train only the new classification head.
    Phase 2 (backbone unfrozen): Fine-tune all layers with a low learning rate.

Output:
    models/mobilenetv2_plantvillage.pth  — best checkpoint
    models/classes.json                   — class index mapping

Swapping to a different model:
    Change build_model() to return any torchvision model.
    Update the input transforms if the model requires different resolution.
    Everything else (training loop, checkpointing, export) stays the same.

Swapping to a different dataset:
    Organize your images as one subfolder per class under --data_dir.
    ImageFolder handles the rest automatically.
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dataset import get_dataloaders, get_class_weights, save_class_mapping

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    MobileNetV2 with a custom classification head.

    MobileNetV2 advantages for this task:
    - Lightweight: 3.4M parameters, runs well on CPU
    - Excellent TFLite/ONNX export support for on-device inference
    - ImageNet pretrained weights provide strong leaf-texture features
    - Final layer size (1280) is a good feature dimension for plant diseases

    To swap architecture (e.g., EfficientNet-B0):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model   = models.mobilenet_v2(weights=weights)

    # Replace 1000-class ImageNet head with our task head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool):
    """Freeze or unfreeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = not frozen

    status = 'FROZEN' if frozen else 'UNFROZEN'
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone {status}. Trainable parameters: {n_trainable:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int) -> float:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += inputs.size(0)

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(loader)}]  "
                f"Loss: {loss.item():.4f}  Acc: {correct/total:.3f}"
            )

    return correct / total


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += inputs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, val_acc, class_names, path):
    torch.save({
        'epoch':            epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc':     val_acc,
        'num_classes':      len(class_names),
        'class_names':      class_names,
    }, path)
    logger.info(f"✓ Checkpoint saved → {path}  (val_acc={val_acc:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        val_split   = args.val_split,
        num_workers = args.workers,
    )
    n_classes = len(class_names)
    logger.info(f"Classes: {n_classes}")

    save_class_mapping(class_names, args.classes_path)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(n_classes, pretrained=True).to(device)

    # Class-weighted loss for imbalanced datasets
    import torchvision.datasets as ds_module
    full_ds = ds_module.ImageFolder(args.data_dir)
    weights  = get_class_weights(full_ds, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    best_val_acc = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1: Train head only (backbone frozen)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*55)
    logger.info("Phase 1: Training classification head (backbone frozen)")
    logger.info("="*55)

    set_backbone_frozen(model, frozen=True)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)

    for epoch in range(1, args.phase1_epochs + 1):
        t0       = time.time()
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"[Phase1 E{epoch:02d}] train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  "
            f"time={time.time()-t0:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, class_names, args.output_path)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Fine-tune all layers (backbone unfrozen)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*55)
    logger.info("Phase 2: Fine-tuning full network (backbone unfrozen)")
    logger.info("="*55)

    set_backbone_frozen(model, frozen=False)
    optimizer = optim.Adam(model.parameters(), lr=args.phase2_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

    for epoch in range(1, args.phase2_epochs + 1):
        t0       = time.time()
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"[Phase2 E{epoch:02d}] train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  "
            f"time={time.time()-t0:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, class_names, args.output_path)

    logger.info(f"\n✓ Training complete. Best val_acc = {best_val_acc:.4f}")
    logger.info(f"  Model saved to: {args.output_path}")
    logger.info(f"  Classes saved to: {args.classes_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Set DEMO_MODE=false in your .env file")
    logger.info(f"  2. Run: python run.py")
    logger.info(f"  3. Export for offline inference: python training/export_onnx.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CropGuard disease classifier')
    parser.add_argument('--data_dir',     default='data/plantvillage',          help='Path to PlantVillage dataset')
    parser.add_argument('--output_path',  default='models/mobilenetv2_plantvillage.pth', help='Where to save the model')
    parser.add_argument('--classes_path', default='models/classes.json',        help='Where to save class names')
    parser.add_argument('--phase1_epochs', type=int, default=5,                 help='Phase 1 (head only) epochs')
    parser.add_argument('--phase2_epochs', type=int, default=15,                help='Phase 2 (fine-tune) epochs')
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--phase1_lr',    type=float, default=1e-3)
    parser.add_argument('--phase2_lr',    type=float, default=1e-4)
    parser.add_argument('--val_split',    type=float, default=0.2)
    parser.add_argument('--workers',      type=int,   default=4)
    args = parser.parse_args()
    main(args)
