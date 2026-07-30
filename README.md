# CropGuard AI

**AI-powered crop disease detection and agricultural intelligence system.**

> Upload a leaf photo → get crop & disease identification, progression forecast, yield impact estimate, and preventive recommendations — with offline support.

---

## Project Structure

```
cropguard/
├── app/                    # Flask application
│   ├── config.py           # All config vars incl. CONFIDENCE_THRESHOLD
│   ├── routes/             # API blueprints (predict, dashboard, analysis)
│   ├── services/
│   │   ├── classifier.py   # MobileNetV2 (demo mode until training/train.py is run)
│   │   ├── confidence.py   # Mandatory abstention utility — check_confidence(score)
│   │   ├── progression.py  # GradientBoostingClassifier — 4000 synthetic samples  ✅ REAL
│   │   ├── yield_model.py  # Ridge regression — 2000 synthetic samples             ✅ REAL
│   │   ├── outbreak.py     # DBSCAN clustering with haversine metric                ✅ REAL
│   │   └── recommendations.py
│   ├── models/             # SQLAlchemy models (Detection, Alert, User)
│   ├── templates/          # index.html (SPA)
│   └── static/             # CSS, JS, Service Worker, manifest
├── training/               # ML training pipeline
│   ├── train.py            # Two-phase MobileNetV2 fine-tuning
│   ├── dataset.py          # PlantVillage DataLoader + augmentation
│   ├── evaluate.py         # Per-class precision/recall/F1
│   └── export_onnx.py      # ONNX export for offline inference
├── tests/
│   ├── conftest.py         # Shared pytest fixtures
│   └── test_cropguard.py   # 49 unit + integration tests
├── models/                 # Saved weights (git-ignored)
├── data/                   # Training data (git-ignored)
├── uploads/                # User images (git-ignored)
├── Dockerfile              # Multi-stage production image
├── docker-compose.yml      # Flask + PostgreSQL
├── wsgi.py                 # gunicorn / waitress entry point
├── demo_seed.py            # Seeds 30 days of realistic detection data
├── pyproject.toml          # pytest config
├── requirements.txt
├── run.py                  # Development entry point
└── README.md
```

---

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
# Install base packages
pip install -r requirements.txt

# Install PyTorch (CPU-only — recommended for dev)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR with GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Configure environment

Create a `.env` file (copy from example below):

```env
FLASK_DEBUG=true
DEMO_MODE=true
CONFIDENCE_THRESHOLD=0.85
SECRET_KEY=your-secret-key-change-in-production
# DATABASE_URL=postgresql://user:pass@host/dbname  (for PostgreSQL)
```

### 4. Run the app

```bash
python run.py
```

Open **http://localhost:5000** — the app starts in **Demo Mode** with a mock classifier.

---

## Training the Real Model

### Step 1: Download the PlantVillage dataset

```bash
# Option A — Kaggle CLI
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/plantvillage

# Option B — TensorFlow Datasets
pip install tensorflow-datasets
python -c "import tensorflow_datasets as tfds; tfds.load('plant_village', split='train', as_supervised=True)"
```

Dataset structure expected:
```
data/plantvillage/
    Apple___Apple_scab/
        image001.jpg
        ...
    Tomato___Late_blight/
        ...
```

### Step 2: Train

```bash
# Default settings (5 epochs head + 15 epochs fine-tune)
python training/train.py --data_dir data/plantvillage

# Custom settings
python training/train.py \
    --data_dir      data/plantvillage \
    --phase1_epochs 5  \
    --phase2_epochs 20 \
    --batch_size    32 \
    --output_path   models/mobilenetv2_plantvillage.pth
```

Training takes approximately:
- **CPU**: 3–6 hours for 20 epochs on PlantVillage (~87K images)
- **GPU (T4)**: ~45 minutes for 20 epochs

### Step 3: Evaluate

```bash
python training/evaluate.py \
    --model_path models/mobilenetv2_plantvillage.pth \
    --data_dir   data/plantvillage
```

### Step 4: Switch to real model

Edit `.env`:
```env
DEMO_MODE=false
```

Restart: `python run.py`

### Step 5: Export for offline inference (ONNX)

```bash
python training/export_onnx.py \
    --model_path  models/mobilenetv2_plantvillage.pth \
    --output_path models/cropguard.onnx \
    --verify
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Submit leaf image for analysis |
| GET | `/api/detections` | List detection history |
| GET | `/api/timeline` | Aggregated timeline for charts |
| GET | `/api/detections/:id` | Single detection detail |
| GET | `/api/progression/:id` | Disease progression forecast (Phase 2) |
| GET | `/api/recommendations/:id` | Preventive actions (Phase 2) |
| GET | `/api/alerts?lat=&lng=` | Nearby outbreak alerts (Phase 3) |
| GET | `/api/yield/:id` | Yield/financial impact estimate (Phase 3) |
| GET | `/health` | Health check |

### POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@leaf_photo.jpg" \
  -F "lat=28.6139" \
  -F "lng=77.2090"
```

**Success response (confidence ≥ 85%):**
```json
{
  "status":       "success",
  "detection_id": 42,
  "crop":         "Tomato",
  "disease":      "Early Blight",
  "is_healthy":   false,
  "confidence":   0.9123,
  "severity":     "moderate",
  "timestamp":    "2026-07-30T17:45:12",
  "demo_mode":    true
}
```

**Low-confidence response (confidence < 85%):**
```json
{
  "status":     "low_confidence",
  "confidence": 0.7234,
  "threshold":  0.85,
  "message":    "The model is not confident enough...",
  "tips":       ["Ensure good natural lighting...", "..."]
}
```

---

## Configuration

All tuneable parameters are in `app/config.py` and overridable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.85` | Abstention threshold (0.0–1.0) |
| `DEMO_MODE` | `true` | Use mock classifier (set false after training) |
| `MODEL_PATH` | `models/mobilenetv2_plantvillage.pth` | Path to trained weights |
| `DATABASE_URL` | SQLite | Set to PostgreSQL URL for production |
| `OUTBREAK_RADIUS_KM` | `50.0` | Alert radius for outbreak detection |
| `OUTBREAK_WINDOW_DAYS` | `7` | Time window for clustering |

---

## Service Status

| Service | Status | Implementation |
|---------|--------|----------------|
| CNN classifier | Demo mode (mock hash) | Run `training/train.py` on PlantVillage to switch |
| Disease progression | ✅ **Real** GradientBoostingClassifier | 4000 synthetic epidemiological samples |
| Yield/financial model | ✅ **Real** Ridge regression | 2000 synthetic FAO-approximated samples |
| Outbreak detection | ✅ **Real** DBSCAN | Haversine metric, auto-persists Alert records |
| Market prices | Hardcoded table | Replace with live commodity price API |
| Soil data | User input | Connect IoT sensors or soil-map API |

> All demo/stub responses include `"stub": true` in JSON. Real model responses have `"stub": false`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output:
```
49 passed in ~11s
```

Test categories:
- **TestConfidenceUtility** — abstention logic (9 tests)
- **TestClassifier** — demo mode predictions (6 tests)
- **TestProgressionModel** — GradientBoosting ML (10 tests)
- **TestYieldModel** — Ridge regression ML (8 tests)
- **TestHaversine** — geo distance utility (4 tests)
- **TestFlaskRoutes** — API integration via Flask test client (12 tests)

---

## Docker Deployment

```bash
# Build and run with PostgreSQL
docker-compose up --build
# → Flask on http://localhost:8000
# → PostgreSQL on :5432 (persistent volume)
```

For production, set these environment variables before running:
```env
SECRET_KEY=a-long-random-secret
POSTGRES_PASSWORD=a-strong-db-password
DEMO_MODE=false
```

## Vercel Deployment

Set environment variables in the Vercel dashboard:
- `DEMO_MODE=false`
- `DATABASE_URL=postgresql://...`
- `MODEL_PATH=<path or presigned URL>`

---

## Model Architecture

**MobileNetV2** was chosen for:
- ✅ Lightweight (3.4M params) — runs well on CPU
- ✅ Excellent ONNX/TFLite export for on-device inference
- ✅ Strong ImageNet pretrained features for leaf texture
- ✅ Proven accuracy on PlantVillage (>95% val accuracy achievable)

To swap to EfficientNet-B0 or other architectures, change only `training/train.py::build_model()`.

---

## License

MIT — see LICENSE for details.
