# ============================================================
# RAG Backend - Dockerfile
# ============================================================
# Multi-stage build for smaller production image

FROM python:3.11-slim AS builder

WORKDIR /build

# Switch to Aliyun apt mirror for faster downloads in China
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    || sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list ; \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    || sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list ; true

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    -r requirements.txt


# ---- Production Stage ----
FROM python:3.11-slim

WORKDIR /app

# Switch to Aliyun apt mirror for faster downloads in China
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    || sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list ; \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    || sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list ; true

# Install runtime dependencies only
# - tesseract: OCR support
# - libgl1 + libglib2.0: OpenCV headless runtime
# - curl: health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Install spaCy model from local .whl file (avoids downloading from GitHub at build time)
COPY en_core_web_sm-3.8.0-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/en_core_web_sm-3.8.0-py3-none-any.whl \
    && rm /tmp/en_core_web_sm-3.8.0-py3-none-any.whl

# Copy backend source code
COPY . /app/backend

# Ensure Python can find the 'backend' package
ENV PYTHONPATH=/app
# Disable Python output buffering (important for Docker logging)
ENV PYTHONUNBUFFERED=1

# Create data directories
RUN mkdir -p /app/data /app/chroma_db /app/temp/vector_uploads /app/logs

EXPOSE 5000

# Health check — 延长 start_period 给 worker 足够的初始化时间
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Production: gunicorn + uvicorn workers
# - workers: 2 适合双核4GB服务器
# - timeout: 300s 兼容 LLM API 慢响应 + 服务器初始化慢
CMD ["gunicorn", "backend.app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "300", \
     "--graceful-timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
