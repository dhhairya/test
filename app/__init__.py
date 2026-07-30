"""
CropGuard AI — Flask Application Factory
"""
import os
import logging
from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# Extension instances (initialized in create_app)
db = SQLAlchemy()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


def create_app(config_name: str = None):
    """
    Application factory.
    Usage:
        app = create_app()                  # uses Config (dev)
        app = create_app('production')      # uses ProductionConfig
    """
    from .config import config_by_name, Config

    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')

    config_class = config_by_name.get(config_name, Config)

    app = Flask(
        __name__,
        static_folder='static',
        template_folder='templates'
    )
    app.config.from_object(config_class)

    # ------------------------------------------------------------------
    # Ensure required directories exist
    # ------------------------------------------------------------------
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(app.root_path), 'models'), exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize extensions
    # ------------------------------------------------------------------
    db.init_app(app)
    CORS(app)  # Allow all origins in dev; restrict origins via env var in production

    # ------------------------------------------------------------------
    # Register blueprints
    # ------------------------------------------------------------------
    from .routes.predict import predict_bp
    from .routes.dashboard import dashboard_bp
    from .routes.progression import progression_bp
    from .routes.recommendations import recommendations_bp
    from .routes.alerts import alerts_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(progression_bp)
    app.register_blueprint(recommendations_bp)
    app.register_blueprint(alerts_bp)

    # ------------------------------------------------------------------
    # Serve Service Worker from root scope (required for full-app PWA)
    # ------------------------------------------------------------------
    @app.route('/sw.js')
    def service_worker():
        return send_from_directory(app.static_folder, 'sw.js',
                                   mimetype='application/javascript')

    @app.route('/manifest.json')
    def manifest():
        return send_from_directory(app.static_folder, 'manifest.json',
                                   mimetype='application/json')

    # ------------------------------------------------------------------
    # Health check endpoint
    # ------------------------------------------------------------------
    @app.route('/health')
    def health():
        from flask import jsonify
        return jsonify({
            'status': 'ok',
            'demo_mode': app.config.get('DEMO_MODE', True),
            'confidence_threshold': app.config.get('CONFIDENCE_THRESHOLD', 0.85)
        })

    # ------------------------------------------------------------------
    # Create DB tables
    # ------------------------------------------------------------------
    with app.app_context():
        db.create_all()

        # Load classifier (non-blocking — demo mode if no weights)
        from .services.classifier import load_classifier
        load_classifier(app)

    return app
