# Multi-stage production container for EduRisk AI (Hugging Face Spaces & Cloud Compatible)
FROM python:3.11-slim AS base

# Install system runtime & build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces non-root user setup (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib_cache \
    PORT=7860

WORKDIR $HOME/app

# Install Python dependencies first for caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project code, trained models, data, and configs
COPY --chown=user:user . .

# Expose default Hugging Face Spaces port (7860) and standard ports (8501, 8000)
EXPOSE 7860 8501 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/_stcore/health || exit 1

# Start command: adapts dynamically to $PORT (7860 for HF Spaces, dynamic on Render, or defaults to 7860)
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-7860} --server.address=0.0.0.0"]
