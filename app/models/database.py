"""
CropGuard AI — SQLAlchemy Database Models

Designed for SQLite (dev) with straightforward migration path to PostgreSQL:
  - No SQLite-specific types used
  - All Float/String/Integer/Boolean/DateTime — all portable
  - To migrate: set DATABASE_URL=postgresql://... and run `flask db upgrade`
"""
from datetime import datetime
from .. import db


class User(db.Model):
    """
    Represents a farmer / app user.
    In Phase 1 MVP we use a single default user (id=1).
    Multi-user auth is a Phase 2+ concern.
    """
    __tablename__ = 'users'

    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(120), nullable=False, default='Default Farmer')
    email       = db.Column(db.String(200), unique=True, nullable=True)
    location    = db.Column(db.String(200), nullable=True)
    soil_type   = db.Column(db.String(50), default='loamy')  # loamy/sandy/clay/silty/peaty
    lat         = db.Column(db.Float, nullable=True)
    lng         = db.Column(db.Float, nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    detections  = db.relationship('Detection', backref='user', lazy='dynamic')

    def to_dict(self):
        return {
            'id':         self.id,
            'name':       self.name,
            'location':   self.location,
            'soil_type':  self.soil_type,
            'lat':        self.lat,
            'lng':        self.lng,
            'created_at': self.created_at.isoformat(),
        }


class Detection(db.Model):
    """
    One crop/disease detection event.
    Stores the image, result, confidence, location, and timestamp.
    """
    __tablename__ = 'detections'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    # Image
    image_path      = db.Column(db.String(500), nullable=True)
    image_filename  = db.Column(db.String(200), nullable=True)

    # Classification result
    crop            = db.Column(db.String(100), nullable=True)   # None if low-confidence
    disease         = db.Column(db.String(200), nullable=True)   # None if low-confidence
    raw_class       = db.Column(db.String(200), nullable=True)   # Original PlantVillage class
    confidence      = db.Column(db.Float, nullable=False)
    is_healthy      = db.Column(db.Boolean, default=False)
    is_low_confidence = db.Column(db.Boolean, default=False)     # abstention flag

    # Severity (early / moderate / severe / none)
    severity        = db.Column(db.String(20), nullable=True)

    # Location
    lat             = db.Column(db.Float, nullable=True)
    lng             = db.Column(db.Float, nullable=True)
    location_name   = db.Column(db.String(200), nullable=True)

    # Metadata
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    synced          = db.Column(db.Boolean, default=True)   # False = pending offline sync
    demo_mode       = db.Column(db.Boolean, default=False)  # was this from demo model?

    def to_dict(self):
        return {
            'id':               self.id,
            'user_id':          self.user_id,
            'crop':             self.crop,
            'disease':          self.disease,
            'confidence':       round(self.confidence, 4) if self.confidence else None,
            'is_healthy':       self.is_healthy,
            'is_low_confidence': self.is_low_confidence,
            'severity':         self.severity,
            'lat':              self.lat,
            'lng':              self.lng,
            'location_name':    self.location_name,
            'timestamp':        self.timestamp.isoformat(),
            'image_filename':   self.image_filename,
            'demo_mode':        self.demo_mode,
        }

    def __repr__(self):
        return f'<Detection id={self.id} crop={self.crop} disease={self.disease} conf={self.confidence:.2f}>'


class Alert(db.Model):
    """
    Regional disease outbreak alert (Phase 3).
    Created by the outbreak detection service when DBSCAN finds a cluster.
    """
    __tablename__ = 'alerts'

    id              = db.Column(db.Integer, primary_key=True)
    disease         = db.Column(db.String(200), nullable=False)
    crop            = db.Column(db.String(100), nullable=True)
    lat             = db.Column(db.Float, nullable=False)
    lng             = db.Column(db.Float, nullable=False)
    radius_km       = db.Column(db.Float, default=50.0)
    severity_level  = db.Column(db.String(20))  # low / medium / high
    detection_count = db.Column(db.Integer, default=1)
    active          = db.Column(db.Boolean, default=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at      = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            'id':              self.id,
            'disease':         self.disease,
            'crop':            self.crop,
            'lat':             self.lat,
            'lng':             self.lng,
            'radius_km':       self.radius_km,
            'severity_level':  self.severity_level,
            'detection_count': self.detection_count,
            'active':          self.active,
            'created_at':      self.created_at.isoformat(),
            'expires_at':      self.expires_at.isoformat() if self.expires_at else None,
        }


class WeatherCache(db.Model):
    """
    Cache for Open-Meteo weather API responses (Phase 2).
    Prevents hammering the external API and supports offline dashboard rendering.
    """
    __tablename__ = 'weather_cache'

    id          = db.Column(db.Integer, primary_key=True)
    lat         = db.Column(db.Float, nullable=False)
    lng         = db.Column(db.Float, nullable=False)
    data_json   = db.Column(db.Text, nullable=False)    # JSON string of forecast data
    fetched_at  = db.Column(db.DateTime, default=datetime.utcnow)

    # Composite index for fast geo lookups
    __table_args__ = (
        db.Index('ix_weather_cache_lat_lng', 'lat', 'lng'),
    )
