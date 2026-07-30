# tests/conftest.py
# Shared pytest fixtures and configuration for CropGuard AI tests.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Use in-memory SQLite for all tests — never touch the real DB
os.environ.setdefault('TESTING',     'true')
os.environ.setdefault('DEMO_MODE',   'true')
os.environ.setdefault('FLASK_DEBUG', 'false')
os.environ.setdefault('SECRET_KEY',  'test-secret')
