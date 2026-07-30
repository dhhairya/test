"""
CropGuard AI — ONNX Export Script

Exports the trained MobileNetV2 checkpoint to ONNX format for:
- On-device (offline) inference via onnxruntime
- TFLite conversion for Android/iOS apps
- Browser-side inference via ONNX Runtime Web

Usage:
    python training/export_onnx.py \
        --model_path models/mobilenetv2_plantvillage.pth \
        --output_path models/cropguard.onnx \
        --classes_path models/classes.json

After export, verify with:
    python training/export_onnx.py --verify --output_path models/cropguard.onnx

To convert to TFLite (requires onnx-tf):
    pip install onnx-tf
    python -c "
    import onnx
    from onnx_tf.backend import prepare
    model = onnx.load('models/cropguard.onnx')
    tf_rep = prepare(model)
    tf_rep.export_graph('models/cropguard_tf')
    "
    # Then use TFLiteConverter on the saved model
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, classes_path: str):
    """Load model and class names from checkpoint."""
    with open(classes_path) as f:
        class_names = json.load(f)

    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, len(class_names)),
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    model.eval()

    return model, class_names


def export_onnx(
    model,
    output_path: str,
    input_shape=(1, 3, 224, 224),
    opset_version: int = 17,
):
    """Export PyTorch model to ONNX."""
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params  = True,
        opset_version  = opset_version,
        do_constant_folding = True,
        input_names    = ['input'],
        output_names   = ['output'],
        dynamic_axes   = {
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        verbose = False,
    )
    logger.info(f"✓ ONNX model exported → {output_path}")


def verify_onnx(model_path: str, onnx_path: str, classes_path: str):
    """Cross-validate PyTorch and ONNX outputs on a random input."""
    import numpy as np
    import onnxruntime as ort

    model, class_names = load_model(model_path, classes_path)

    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        pt_out = torch.softmax(model(dummy), dim=1).numpy()

    session  = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out  = session.run(None, {'input': dummy.numpy()})[0]
    ort_prob = torch.softmax(torch.tensor(ort_out), dim=1).numpy()

    max_diff = float(abs(pt_out - ort_prob).max())
    logger.info(f"PyTorch vs ONNX max output difference: {max_diff:.6f}")

    if max_diff < 1e-4:
        logger.info("✓ ONNX export verified — outputs match within tolerance.")
    else:
        logger.warning(f"⚠ Output mismatch ({max_diff:.6f}). Check model carefully.")

    pt_class   = class_names[pt_out.argmax()]
    ort_class  = class_names[ort_prob.argmax()]
    logger.info(f"PyTorch top class: {pt_class}")
    logger.info(f"ONNX    top class: {ort_class}")


def main(args):
    logger.info(f"Loading model from {args.model_path}")
    model, class_names = load_model(args.model_path, args.classes_path)
    logger.info(f"Loaded {len(class_names)} classes")

    export_onnx(model, args.output_path, opset_version=args.opset)

    # Validate ONNX model structure
    onnx_model = onnx.load(args.output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✓ ONNX model structure check passed")

    if args.verify:
        verify_onnx(args.model_path, args.output_path, args.classes_path)

    size_mb = Path(args.output_path).stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model size: {size_mb:.1f} MB")
    logger.info("\nDeploy this ONNX model with:")
    logger.info("  import onnxruntime as ort")
    logger.info(f"  session = ort.InferenceSession('{args.output_path}')")
    logger.info("  output  = session.run(None, {'input': preprocessed_image})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export CropGuard model to ONNX')
    parser.add_argument('--model_path',   default='models/mobilenetv2_plantvillage.pth')
    parser.add_argument('--output_path',  default='models/cropguard.onnx')
    parser.add_argument('--classes_path', default='models/classes.json')
    parser.add_argument('--opset',        type=int, default=17)
    parser.add_argument('--verify',       action='store_true', help='Cross-validate outputs')
    args = parser.parse_args()
    main(args)
