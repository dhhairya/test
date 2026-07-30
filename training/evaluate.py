"""
CropGuard AI — Model Evaluation Script

Generates per-class precision/recall/F1, confusion matrix,
and overall accuracy report on a held-out test set.

Usage:
    python training/evaluate.py \
        --model_path models/mobilenetv2_plantvillage.pth \
        --data_dir   data/plantvillage_test \
        --classes_path models/classes.json

Or evaluate on the validation split:
    python training/evaluate.py --use_val_split
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(model_path, num_classes, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, num_classes),
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.classes_path) as f:
        class_names = json.load(f)

    model = load_model(args.model_path, len(class_names), device)

    dataset = datasets.ImageFolder(args.data_dir, transform=VAL_TRANSFORMS)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    logger.info(f"Evaluating on {len(dataset)} images from {args.data_dir}")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    logger.info(f"\n{'='*60}")
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"{'='*60}")

    # Per-class report
    print("\nPer-class Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=3
    ))

    # Top-5 most confused classes
    cm = confusion_matrix(all_labels, all_preds)
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and off_diag[i, j] > 0:
                confused_pairs.append((off_diag[i, j], class_names[i], class_names[j]))

    confused_pairs.sort(reverse=True)
    print("\nTop-10 confused class pairs (true → predicted: count):")
    for count, true_cls, pred_cls in confused_pairs[:10]:
        print(f"  {true_cls:45s} → {pred_cls}: {count}")

    if args.save_report:
        report = {
            'accuracy': float(accuracy),
            'num_classes': len(class_names),
            'num_samples': len(dataset),
            'confused_pairs': [(int(c), t, p) for c, t, p in confused_pairs[:20]],
        }
        out = Path('models/eval_report.json')
        out.write_text(json.dumps(report, indent=2))
        logger.info(f"Report saved to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',   default='models/mobilenetv2_plantvillage.pth')
    parser.add_argument('--classes_path', default='models/classes.json')
    parser.add_argument('--data_dir',     default='data/plantvillage')
    parser.add_argument('--batch_size',   type=int, default=64)
    parser.add_argument('--workers',      type=int, default=4)
    parser.add_argument('--save_report',  action='store_true')
    args = parser.parse_args()
    evaluate(args)
