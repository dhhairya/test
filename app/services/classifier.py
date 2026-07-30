"""
CropGuard AI — Crop & Disease Classifier Service

Wraps MobileNetV2 (pretrained on ImageNet, fine-tuned on PlantVillage).

DEMO MODE (default until trained):
    Returns deterministic-but-varied predictions based on image statistics.
    The confidence score varies realistically so the full confidence-abstention
    pipeline can be exercised in development.

PRODUCTION MODE (after running training/train.py):
    Set DEMO_MODE=false in .env — the service loads the real checkpoint
    transparently with no other code changes.

Swapping to a different model architecture:
    1. Change _build_model() to return your new architecture.
    2. Update training/train.py accordingly.
    3. Re-export ONNX via training/export_onnx.py.
    4. All other code stays the same.
"""
import os
import io
import json
import logging
import hashlib
from typing import Dict, Any, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PlantVillage class list (38 classes, 14 crop types)
# Source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# The training pipeline generates models/classes.json with the same ordering.
# ──────────────────────────────────────────────────────────────────────────────
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Human-readable disease descriptions for the UI
DISEASE_DESCRIPTIONS = {
    "Apple_scab": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
    "Black_rot": "Fungal infection causing circular lesions with dark borders.",
    "Cedar_apple_rust": "Fungal disease producing orange rust-colored spots.",
    "Powdery_mildew": "White powdery fungal coating on leaf surfaces.",
    "Cercospora_leaf_spot_Gray_leaf_spot": "Gray-brown rectangular lesions between leaf veins.",
    "Common_rust": "Orange-red pustules scattered on both leaf surfaces.",
    "Northern_Leaf_Blight": "Large, cigar-shaped gray-green lesions on leaves.",
    "Esca_(Black_Measles)": "Causes interveinal chlorosis and necrosis in grapevines.",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Dark brown irregular spots, often with yellow halos.",
    "Haunglongbing_(Citrus_greening)": "Bacterial disease causing yellowing and misshapen fruit.",
    "Bacterial_spot": "Small, water-soaked spots that turn dark and angular.",
    "Early_blight": "Concentric ring lesions (target-board pattern) on older leaves.",
    "Late_blight": "Water-soaked lesions turning brown-black, often with white mold.",
    "Leaf_Mold": "Yellow spots on upper leaf surface, olive-green mold below.",
    "Septoria_leaf_spot": "Small circular spots with dark borders and gray centers.",
    "Spider_mites_Two-spotted_spider_mite": "Tiny mites causing stippled, yellowing leaves.",
    "Target_Spot": "Concentric ring lesions resembling a target.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Viral disease causing yellowing, curling, and stunted growth.",
    "Tomato_mosaic_virus": "Mosaic pattern of light/dark green on leaves.",
    "Leaf_scorch": "Brown, scorched leaf margins and tips.",
}


def parse_class_name(class_name: str) -> Dict[str, Any]:
    """
    Parse a PlantVillage class label into structured components.

    Example:
        'Tomato___Early_blight'  →  {crop: 'Tomato', disease: 'Early Blight', is_healthy: False}
        'Apple___healthy'        →  {crop: 'Apple',  disease: 'Healthy',       is_healthy: True}
    """
    parts = class_name.split('___')
    crop_raw = parts[0]
    disease_raw = parts[1] if len(parts) > 1 else 'Unknown'

    # Clean crop name
    crop = (crop_raw
            .replace('_', ' ')
            .replace('(including sour)', '')
            .replace(',', '')
            .strip()
            .title())

    is_healthy = disease_raw.lower() == 'healthy'

    if is_healthy:
        disease = 'Healthy'
        description = 'No disease detected. Crop appears healthy.'
    else:
        disease = disease_raw.replace('_', ' ').strip()
        description = DISEASE_DESCRIPTIONS.get(disease_raw, '')

    return {
        'crop':        crop,
        'disease':     disease,
        'is_healthy':  is_healthy,
        'raw_class':   class_name,
        'description': description,
    }


def validate_image(file_bytes: bytes) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validate uploaded image bytes.

    Returns:
        (is_valid, error_message, pil_image)
        - is_valid=True  → pil_image is a valid PIL Image ready for inference
        - is_valid=False → error_message describes why validation failed
    """
    if not file_bytes:
        return False, "Uploaded file is empty.", None

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()                             # detects corrupted files
        img = Image.open(io.BytesIO(file_bytes)) # re-open after verify (verify seeks to end)
    except Exception as exc:
        return False, f"Invalid or corrupted image file: {exc}", None

    allowed_formats = ('JPEG', 'PNG', 'WEBP')
    if img.format not in allowed_formats:
        return False, f"Unsupported format '{img.format}'. Upload a JPEG, PNG, or WebP image.", None

    if img.width < 64 or img.height < 64:
        return False, (
            f"Image too small ({img.width}×{img.height}px). "
            "Please upload a higher-resolution photo (min 64×64)."
        ), None

    return True, None, img


# ──────────────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────────────

class CropDiseaseClassifier:
    """
    MobileNetV2-based crop disease classifier.

    The model architecture:
        - Backbone: MobileNetV2 (ImageNet pretrained) — lightweight, exportable to ONNX/TFLite
        - Head:     Linear(1280 → num_classes) replacing the default 1000-class head
        - Input:    224×224 RGB, normalized with ImageNet mean/std
        - Output:   Softmax probabilities over 38 PlantVillage classes

    Training strategy (see training/train.py):
        Phase 1 (5 epochs): Freeze backbone, train only the new head.
        Phase 2 (10 epochs): Unfreeze backbone, fine-tune everything with low LR.
    """

    def __init__(self):
        self.model        = None
        self.classes      = PLANTVILLAGE_CLASSES
        self.device       = None
        self.transform    = None
        self._loaded      = False
        self._demo_mode   = True

    def _init_torch(self):
        """Lazy-load PyTorch to avoid import errors if not installed."""
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        import torchvision.models as models

        self.torch     = torch
        self.nn        = nn
        self.transforms = transforms
        self.tv_models = models
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])

    def _build_model(self, num_classes: int):
        """Build MobileNetV2 with custom classification head."""
        model = self.tv_models.mobilenet_v2(weights=None)
        model.classifier[1] = self.nn.Linear(model.last_channel, num_classes)
        return model

    def load(self, model_path: str, classes_path: str):
        """Load trained model weights from checkpoint."""
        self._init_torch()

        # Load class labels
        if os.path.exists(classes_path):
            with open(classes_path) as f:
                self.classes = json.load(f)
            logger.info(f"Loaded {len(self.classes)} classes from {classes_path}")
        else:
            logger.warning(f"classes.json not found at {classes_path}. Using defaults.")

        self.model = self._build_model(len(self.classes))

        if os.path.exists(model_path):
            checkpoint = self.torch.load(model_path, map_location=self.device, weights_only=True)
            state = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state)
            epoch = checkpoint.get('epoch', '?')
            acc   = checkpoint.get('best_val_acc', '?')
            logger.info(f"✓ Loaded trained weights from {model_path} (epoch {epoch}, val_acc {acc})")
            self._demo_mode = False
        else:
            logger.warning(
                f"No trained weights at {model_path}. "
                "Running in DEMO_MODE — run `python training/train.py` to train."
            )
            self._demo_mode = True

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_demo(self, img: Image.Image) -> Dict[str, Any]:
        """
        Demo-mode prediction.

        Uses a deterministic hash of image pixel statistics to select a class
        and compute a plausible confidence score. The result varies with image
        content so the full confidence/abstention pipeline can be exercised.

        NOTE: This is NOT a trained prediction — it is only for UI/API testing.
        """
        img_small = img.convert('RGB').resize((32, 32))
        pixels    = list(img_small.getdata())
        n         = len(pixels)

        avg_r = sum(p[0] for p in pixels) / n
        avg_g = sum(p[1] for p in pixels) / n
        avg_b = sum(p[2] for p in pixels) / n

        # Deterministic seed from pixel stats
        seed_str = f"{avg_r:.1f}{avg_g:.1f}{avg_b:.1f}"
        seed     = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % len(self.classes)

        selected_class = self.classes[seed]

        # Confidence: greener images → higher confidence (proxy for leaf-like content)
        total      = avg_r + avg_g + avg_b + 1e-6
        greenness  = avg_g / total
        saturation = (max(avg_r, avg_g, avg_b) - min(avg_r, avg_g, avg_b)) / 255
        confidence = 0.60 + (greenness * 0.25) + (saturation * 0.15)
        confidence = min(confidence, 0.99)

        parsed = parse_class_name(selected_class)
        return {
            **parsed,
            'confidence': round(confidence, 4),
            'class_index': seed,
            'demo_mode':   True,
        }

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a PIL Image.
        Falls back to demo mode if model is not loaded.
        """
        if not self._loaded or self._demo_mode:
            return self.predict_demo(img)

        try:
            tensor = self.transform(img.convert('RGB')).unsqueeze(0).to(self.device)

            with self.torch.no_grad():
                logits       = self.model(tensor)
                probs        = self.torch.softmax(logits, dim=1)
                confidence, predicted = probs.max(1)

            class_idx  = predicted.item()
            conf_score = confidence.item()
            class_name = self.classes[class_idx]
            parsed     = parse_class_name(class_name)

            return {
                **parsed,
                'confidence':  round(conf_score, 4),
                'class_index': class_idx,
                'demo_mode':   False,
            }
        except Exception as exc:
            logger.error(f"Inference error: {exc}")
            raise RuntimeError(f"Model inference failed: {exc}") from exc


# ── Module-level singleton ─────────────────────────────────────────────────────

_classifier: Optional[CropDiseaseClassifier] = None


def get_classifier() -> CropDiseaseClassifier:
    """Return the module-level classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = CropDiseaseClassifier()
    return _classifier


def load_classifier(app) -> CropDiseaseClassifier:
    """
    Initialize and optionally load the classifier within the Flask app context.
    Called once from the app factory (app/__init__.py).
    """
    clf       = get_classifier()
    demo_mode = app.config.get('DEMO_MODE', True)

    if not demo_mode:
        model_path   = app.config.get('MODEL_PATH', 'models/mobilenetv2_plantvillage.pth')
        classes_path = app.config.get('CLASSES_PATH', 'models/classes.json')
        try:
            clf.load(model_path, classes_path)
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}. Falling back to demo mode.")
            clf._demo_mode = True
    else:
        logger.info("DEMO_MODE=true — using mock classifier. "
                    "Set DEMO_MODE=false after training to use the real model.")

    return clf
