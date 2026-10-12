# Multi-stage production container for EduRisk AI
FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained artifacts
COPY configs/ ./configs/
COPY data/ ./data/
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/
COPY api/ ./api/
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

# Create non-root system user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose FastAPI (8000) and Streamlit (8501 / 7860 for HF Spaces)
EXPOSE 8000 8501 7860

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib_cache \
    PORT=8501

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default launch command: starts Streamlit dashboard
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
