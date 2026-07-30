"""
CropGuard AI — Demo Seed Script
Populates the database with 30 days of realistic detection history
so the dashboard charts look great on first open.

Usage:
    python demo_seed.py

Safe to re-run: checks if data already exists before inserting.
"""
import sys, os, random, hashlib
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load env before importing app
from dotenv import load_dotenv
load_dotenv()

from app import create_app, db
from app.models.database import Detection

# ── Realistic disease/crop pairs from PlantVillage ───────────────────────────
DISEASE_PROFILES = [
    # (crop, disease, is_healthy, severity, conf_range)
    ("Tomato",      "Tomato___Early_blight",                       False, "early",    (0.87, 0.97)),
    ("Tomato",      "Tomato___Late_blight",                        False, "severe",   (0.88, 0.98)),
    ("Tomato",      "Tomato___Leaf_Mold",                          False, "moderate", (0.86, 0.94)),
    ("Tomato",      "Tomato___healthy",                            True,  "none",     (0.91, 0.99)),
    ("Potato",      "Potato___Early_blight",                       False, "moderate", (0.85, 0.96)),
    ("Potato",      "Potato___Late_blight",                        False, "severe",   (0.89, 0.98)),
    ("Potato",      "Potato___healthy",                            True,  "none",     (0.92, 0.99)),
    ("Apple",       "Apple___Apple_scab",                          False, "early",    (0.86, 0.95)),
    ("Apple",       "Apple___Black_rot",                           False, "moderate", (0.87, 0.96)),
    ("Apple",       "Apple___healthy",                             True,  "none",     (0.90, 0.99)),
    ("Corn",        "Corn_(maize)___Common_rust",                  False, "moderate", (0.85, 0.94)),
    ("Corn",        "Corn_(maize)___Northern_Leaf_Blight",         False, "early",    (0.86, 0.95)),
    ("Corn",        "Corn_(maize)___healthy",                      True,  "none",     (0.93, 0.99)),
    ("Grape",       "Grape___Black_rot",                           False, "severe",   (0.88, 0.97)),
    ("Grape",       "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",  False, "moderate", (0.85, 0.95)),
    ("Pepper,_bell","Pepper,_bell___Bacterial_spot",               False, "early",    (0.86, 0.95)),
    ("Pepper,_bell","Pepper,_bell___healthy",                      True,  "none",     (0.91, 0.99)),
    ("Rice",        "Rice___Brown_spot",                           False, "early",    (0.87, 0.95)),
    ("Rice",        "Rice___Bacterial_leaf_blight",                False, "moderate", (0.88, 0.97)),
    ("Wheat",       "Wheat___Yellow_rust",                         False, "moderate", (0.86, 0.94)),
    ("Wheat",       "Wheat___Powdery_mildew",                      False, "early",    (0.85, 0.93)),
]

LOCATIONS = [
    ("Delhi, India",     28.6139, 77.2090),
    ("Mumbai, India",    19.0760, 72.8777),
    ("Punjab, India",    30.7333, 76.7794),
    ("Nashik, India",    19.9975, 73.7898),
    ("Agra, India",      27.1767, 78.0081),
    ("Ludhiana, India",  30.9010, 75.8573),
]

# ── Fake image path generator (images not actually stored in seed) ────────────
def fake_image_path(idx):
    return f"uploads/seed_{idx:04d}.jpg"


def seed_database(n_days=30, detections_per_day_range=(2, 8)):
    app = create_app()
    with app.app_context():
        existing = Detection.query.count()
        if existing > 10:
            print(f"Database already has {existing} detections. Skipping seed.")
            print("To re-seed: delete cropguard.db and run this script again.")
            return

        print(f"Seeding {n_days} days of detection history...")
        rng = random.Random(42)  # deterministic for reproducibility
        inserted = 0

        for day_offset in range(n_days, 0, -1):
            date = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=day_offset)
            n_detections = rng.randint(*detections_per_day_range)

            for _ in range(n_detections):
                profile = rng.choice(DISEASE_PROFILES)
                crop, class_name, is_healthy, severity, conf_range = profile
                confidence = round(rng.uniform(*conf_range), 4)
                location   = rng.choice(LOCATIONS)
                loc_name, lat, lng = location

                # Vary timestamp within the day
                timestamp = date.replace(
                    hour=rng.randint(6, 19),
                    minute=rng.randint(0, 59),
                    second=rng.randint(0, 59),
                    microsecond=0,
                )

                # Parse crop/disease from class name
                parts   = class_name.split("___")
                disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"

                detection = Detection(
                    image_path    = fake_image_path(inserted),
                    crop          = crop,
                    disease       = disease,
                    raw_class     = class_name,
                    confidence    = confidence,
                    is_healthy    = is_healthy,
                    severity      = severity,
                    location_name = loc_name,
                    lat           = lat,
                    lng           = lng,
                    timestamp     = timestamp,
                    demo_mode     = True,
                )
                db.session.add(detection)
                inserted += 1

        db.session.commit()
        print(f"Done. Inserted {inserted} detections across {n_days} days.")
        print(f"Open http://localhost:5000 and click the Dashboard tab to see the charts.")

        # Summary
        total    = Detection.query.count()
        healthy  = Detection.query.filter_by(is_healthy=True).count()
        diseased = total - healthy
        print(f"\nDatabase summary:")
        print(f"  Total detections : {total}")
        print(f"  Healthy          : {healthy}")
        print(f"  Diseased         : {diseased}")
        print(f"  Health rate      : {int(healthy/total*100)}%")


if __name__ == "__main__":
    seed_database()
