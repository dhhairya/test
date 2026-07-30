"""
CropGuard AI — Entry Point
Run with:  python run.py
Or:        flask run
"""
import sys, os

# Force UTF-8 output on Windows (avoids emoji codec errors in log messages)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

load_dotenv()

from app import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    print(f"\n{'='*55}")
    print(f"  CropGuard AI -- Starting on http://localhost:{port}")
    print(f"  Demo Mode: {app.config.get('DEMO_MODE', True)}")
    print(f"  Confidence Threshold: {app.config.get('CONFIDENCE_THRESHOLD', 0.85):.0%}")
    print(f"{'='*55}\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
