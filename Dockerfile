FROM python:3.12-slim

WORKDIR /app

# Python runtime configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
# gcc + libpq-dev are needed to build psycopg[c] wheels.
# If you switch to psycopg[binary] or psycopg2-binary, you can drop gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install application dependencies (this is what pulls torch — see notes below)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

# Copy application
COPY . .

# Non-root user for security
RUN useradd --create-home --uid 1000 atlas && chown -R atlas:atlas /app
USER atlas

EXPOSE 8000

# Health check for App Runner / ECS
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]