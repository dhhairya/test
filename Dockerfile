# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install PyTorch CPU (slim build — no GPU drivers in container)
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create runtime directories
RUN mkdir -p uploads models data && \
    chmod 755 uploads

# Environment defaults (override via docker run -e or docker-compose)
ENV FLASK_DEBUG=false \
    DEMO_MODE=true \
    CONFIDENCE_THRESHOLD=0.85 \
    PORT=8000

# Non-root user for security
RUN useradd -m -u 1001 cropguard && \
    chown -R cropguard:cropguard /app
USER cropguard

EXPOSE 8000

# Use gunicorn in production
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "wsgi:app"]
