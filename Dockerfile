# Builder stage for dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY bin /root/.local/bin/
ENV PATH="/root/.local/bin/:$PATH"

# Copy requirements
COPY requirements.txt ./

# Create virtualenv and install packages
RUN uv venv /app/.venv && \
    uv pip install --no-cache-dir -r requirements.txt

# Copy local face_recognition_models with actual model files
COPY face_recognition_models /app/face_recognition_models

# Install face_recognition_models from local directory
RUN uv pip install --no-cache-dir /app/face_recognition_models

# Final stage
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY . .

EXPOSE 5000

ENTRYPOINT ["/app/.venv/bin/gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
