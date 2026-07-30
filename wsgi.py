"""
CropGuard AI — WSGI entry point for production servers.

Use with gunicorn:
    gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 2 wsgi:app

Use with uwsgi:
    uwsgi --http :8000 --module wsgi:app --processes 2 --threads 2

Use with Waitress (Windows-compatible):
    pip install waitress
    waitress-serve --host=0.0.0.0 --port=8000 wsgi:app
"""
import sys
import os

# Ensure UTF-8 output on all platforms
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from app import create_app

app = create_app()

if __name__ == '__main__':
    # Direct run: use waitress (cross-platform production server)
    try:
        from waitress import serve
        port = int(os.environ.get('PORT', 8000))
        print(f"Starting waitress production server on http://0.0.0.0:{port}")
        serve(app, host='0.0.0.0', port=port, threads=4)
    except ImportError:
        # Fall back to Flask dev server with a stern warning
        print("WARNING: waitress not installed. Running Flask dev server (NOT for production).")
        print("Install: pip install waitress")
        port = int(os.environ.get('PORT', 8000))
        app.run(host='0.0.0.0', port=port, debug=False)
