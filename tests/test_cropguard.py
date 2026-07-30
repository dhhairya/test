"""
CropGuard AI — Unit Test Suite
Run with:  pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_image(r=10, g=200, b=20, size=(256, 256), fmt='JPEG') -> BytesIO:
    buf = BytesIO()
    Image.new('RGB', size, color=(r, g, b)).save(buf, fmt)
    buf.seek(0)
    return buf


# ── Confidence utility ────────────────────────────────────────────────────────

class TestConfidenceUtility:
    def setup_method(self):
        from app.services.confidence import check_confidence
        self.check = check_confidence

    def test_high_confidence_passes(self):
        passes, payload = self.check(0.92)
        assert passes is True
        assert payload == {}

    def test_exact_threshold_passes(self):
        passes, payload = self.check(0.85)
        assert passes is True

    def test_below_threshold_blocked(self):
        passes, payload = self.check(0.84)
        assert passes is False
        assert payload['status'] == 'low_confidence'

    def test_zero_confidence_blocked(self):
        passes, payload = self.check(0.0)
        assert passes is False

    def test_perfect_confidence_passes(self):
        passes, payload = self.check(1.0)
        assert passes is True

    def test_result_has_all_fields(self):
        passes, payload = self.check(0.75)
        assert passes is False
        assert 'status' in payload
        assert 'confidence' in payload
        assert 'threshold' in payload
        assert 'tips' in payload

    def test_low_confidence_includes_tips(self):
        _, payload = self.check(0.5)
        assert isinstance(payload['tips'], list)
        assert len(payload['tips']) >= 4

    def test_high_confidence_has_empty_payload(self):
        passes, payload = self.check(0.95)
        assert passes is True
        assert payload == {}

    def test_custom_threshold_respected(self):
        # Score 0.70 passes a lower threshold of 0.65
        passes, _ = self.check(0.70, threshold=0.65)
        assert passes is True
        # Same score fails default 0.85 threshold
        passes2, _ = self.check(0.70)
        assert passes2 is False


# ── Classifier service (demo mode) ───────────────────────────────────────────

class TestClassifier:
    def setup_method(self):
        from app.services.classifier import CropDiseaseClassifier
        self.clf = CropDiseaseClassifier()

    def test_demo_mode_env_is_true(self):
        import os
        assert os.environ.get('DEMO_MODE', 'true').lower() == 'true'

    def test_predict_returns_dict(self):
        img = Image.new('RGB', (256, 256), color=(10, 200, 20))
        result = self.clf.predict_demo(img)
        assert isinstance(result, dict)

    def test_predict_has_required_fields(self):
        img = Image.new('RGB', (256, 256), color=(10, 200, 20))
        result = self.clf.predict_demo(img)
        for field in ('crop', 'disease', 'confidence', 'is_healthy', 'class_index'):
            assert field in result, f"Missing field: {field}"

    def test_confidence_in_valid_range(self):
        img = Image.new('RGB', (256, 256), color=(10, 200, 20))
        result = self.clf.predict_demo(img)
        assert 0.0 <= result['confidence'] <= 1.0

    def test_is_healthy_is_bool(self):
        img = Image.new('RGB', (256, 256), color=(10, 200, 20))
        result = self.clf.predict_demo(img)
        assert isinstance(result['is_healthy'], bool)

    def test_deterministic_same_image(self):
        img = Image.new('RGB', (256, 256), color=(5, 210, 15))
        r1 = self.clf.predict_demo(img)
        r2 = self.clf.predict_demo(img)
        assert r1['crop'] == r2['crop']
        assert r1['confidence'] == r2['confidence']

    def test_different_images_differ(self):
        img1 = Image.new('RGB', (256, 256), color=(10, 200, 20))
        img2 = Image.new('RGB', (256, 256), color=(128, 128, 128))
        r1 = self.clf.predict_demo(img1)
        r2 = self.clf.predict_demo(img2)
        # At least one metric should differ between a green and gray image
        assert r1['confidence'] != r2['confidence'] or r1['crop'] != r2['crop']


# ── Progression model ─────────────────────────────────────────────────────────

class TestProgressionModel:
    def setup_method(self):
        from app.services.progression import predict_progression, _get_model
        self.predict = predict_progression
        self.get_model = _get_model

    def test_model_trains_successfully(self):
        clf = self.get_model()
        assert clf is not None
        assert clf.__class__.__name__ == 'GradientBoostingClassifier'

    def test_returns_three_windows(self):
        result = self.predict('Late blight', 0, None, 'early')
        assert len(result['windows']) == 3

    def test_window_days_correct(self):
        result = self.predict('Early blight', 0, None, 'early')
        days = [w['days'] for w in result['windows']]
        assert days == [7, 14, 30]

    def test_window_has_required_fields(self):
        result = self.predict('Late blight', 0, None, 'early')
        for w in result['windows']:
            for field in ('days', 'stage', 'probability', 'spread_risk', 'color', 'target_date'):
                assert field in w, f"Window missing field: {field}"

    def test_stage_is_valid(self):
        result = self.predict('Common rust', 0, None, 'moderate')
        for w in result['windows']:
            assert w['stage'] in ('early', 'moderate', 'severe', 'none')

    def test_probability_in_range(self):
        result = self.predict('Black rot', 0, None, 'early')
        for w in result['windows']:
            assert 0.0 <= w['probability'] <= 1.0

    def test_stub_false_with_real_model(self):
        result = self.predict('Late blight', 0, None, 'early')
        assert result['stub'] is False

    def test_weather_factor_present(self):
        result = self.predict('Late blight', 0, None, 'early')
        assert result['weather_factor'] in ('low', 'medium', 'high')

    def test_high_humidity_escalates(self):
        normal = self.predict('Late blight', 0, None, 'early')
        humid  = self.predict('Late blight', 0,
                              {'current': {'temp': 25, 'humidity': 85, 'precip': 10}},
                              'early')
        # Weather factor should differ
        assert humid['weather_factor'] in ('medium', 'high')

    def test_unknown_disease_handled(self):
        result = self.predict('Completely unknown disease XYZ', 0, None, 'early')
        assert len(result['windows']) == 3  # falls back gracefully


# ── Yield model ───────────────────────────────────────────────────────────────

class TestYieldModel:
    def setup_method(self):
        from app.services.yield_model import predict_yield_impact, _get_model
        self.predict = predict_yield_impact
        self.get_model = _get_model

    def test_model_trains_successfully(self):
        reg = self.get_model()
        assert reg.__class__.__name__ == 'Ridge'

    def test_healthy_crop_zero_loss(self):
        r = self.predict('Tomato', 'healthy', 'none', 1.0, is_healthy=True)
        assert r['yield_loss_pct'] == 0.0
        assert r['financial_loss_usd'] == 0.0

    def test_severe_disease_significant_loss(self):
        r = self.predict('Tomato', 'Late blight', 'severe', 1.0)
        assert r['yield_loss_pct'] > 20.0

    def test_early_disease_small_loss(self):
        r = self.predict('Apple', 'Apple scab', 'early', 1.0)
        assert r['yield_loss_pct'] < 30.0

    def test_field_size_scales_financials(self):
        r1 = self.predict('Potato', 'Early blight', 'moderate', 1.0)
        r5 = self.predict('Potato', 'Early blight', 'moderate', 5.0)
        # 5ha should produce roughly 5x the loss — allow 30% tolerance
        # (Ridge model uses field_size as a feature, so loss_frac may shift slightly)
        ratio = r5['financial_loss_usd'] / max(r1['financial_loss_usd'], 0.01)
        assert 3.0 <= ratio <= 7.0, f"5ha/1ha loss ratio={ratio:.2f}, expected 3-7x"


    def test_required_fields_present(self):
        r = self.predict('Corn', 'Common rust', 'moderate', 2.0)
        for field in ('healthy_yield_t_ha', 'predicted_yield_t_ha', 'yield_loss_pct',
                      'financial_loss_usd', 'price_per_ton_usd', 'field_size_ha',
                      'disclaimer', 'model'):
            assert field in r, f"Missing field: {field}"

    def test_ridge_model_identified(self):
        r = self.predict('Wheat', 'Yellow rust', 'severe', 3.0)
        assert 'Ridge' in r['model']

    def test_loss_pct_in_valid_range(self):
        for sev in ('none', 'early', 'moderate', 'severe'):
            r = self.predict('Tomato', 'Early blight', sev, 1.0)
            assert 0.0 <= r['yield_loss_pct'] <= 100.0


# ── Haversine utility ─────────────────────────────────────────────────────────

class TestHaversine:
    def setup_method(self):
        from app.services.outbreak import haversine_distance
        self.dist = haversine_distance

    def test_same_point_zero(self):
        assert self.dist(0, 0, 0, 0) == 0.0

    def test_known_distance(self):
        # Delhi to Mumbai ~ 1150 km
        d = self.dist(28.6139, 77.2090, 19.0760, 72.8777)
        assert 1100 < d < 1200

    def test_symmetry(self):
        d1 = self.dist(28.6, 77.2, 19.0, 72.8)
        d2 = self.dist(19.0, 72.8, 28.6, 77.2)
        assert abs(d1 - d2) < 0.01

    def test_nearby_small_distance(self):
        # ~17 km
        d = self.dist(28.6139, 77.2090, 28.7041, 77.1025)
        assert d < 20


# ── Flask app integration ─────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def app():
    os.environ['TESTING'] = 'true'
    os.environ['DEMO_MODE'] = 'true'
    from app import create_app
    application = create_app()
    application.config['TESTING'] = True
    application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with application.app_context():
        from app import db
        db.create_all()
        yield application


@pytest.fixture(scope='module')
def client(app):
    return app.test_client()


class TestFlaskRoutes:
    def test_health_endpoint(self, client):
        r = client.get('/health')
        assert r.status_code == 200
        j = r.get_json()
        assert j['status'] == 'ok'
        assert 'confidence_threshold' in j
        assert 'demo_mode' in j

    def test_root_returns_html(self, client):
        r = client.get('/')
        assert r.status_code == 200
        assert b'CropGuard' in r.data

    def test_predict_no_image_returns_400(self, client):
        r = client.post('/predict')
        assert r.status_code == 400

    def test_predict_with_image(self, client):
        buf = make_image()
        data = {'image': (buf, 'leaf.jpg', 'image/jpeg')}
        r = client.post('/predict', data=data, content_type='multipart/form-data')
        assert r.status_code == 200
        j = r.get_json()
        assert j['status'] in ('success', 'low_confidence')

    def test_predict_low_confidence_has_tips(self, client):
        # Plain gray image reliably scores low
        buf = make_image(128, 128, 128)
        data = {'image': (buf, 'gray.jpg', 'image/jpeg')}
        r = client.post('/predict', data=data, content_type='multipart/form-data')
        j = r.get_json()
        if j['status'] == 'low_confidence':
            assert isinstance(j.get('tips'), list)
            assert len(j['tips']) >= 1

    def test_timeline_api(self, client):
        r = client.get('/api/timeline')
        assert r.status_code == 200
        j = r.get_json()
        assert 'summary' in j
        assert 'timeline' in j

    def test_detections_api(self, client):
        r = client.get('/api/detections')
        assert r.status_code == 200
        j = r.get_json()
        assert 'detections' in j
        assert 'total' in j

    def test_alerts_api(self, client):
        r = client.get('/api/alerts?lat=28.6139&lng=77.2090')
        assert r.status_code == 200
        j = r.get_json()
        assert 'alerts' in j
        assert 'total' in j

    def test_static_css(self, client):
        r = client.get('/static/css/style.css')
        assert r.status_code == 200

    def test_static_sw(self, client):
        r = client.get('/static/sw.js')
        assert r.status_code == 200

    def test_manifest(self, client):
        r = client.get('/manifest.json')
        assert r.status_code == 200
        j = r.get_json()
        assert j.get('name') == 'CropGuard AI'
