# NEXUS v1 — single-container deployment (Render / Railway / Fly.io)
FROM python:3.11-slim

# git is required at runtime (GitPython clones target repositories)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend

# Non-root user
RUN useradd -m nexus && chown -R nexus /app
USER nexus

EXPOSE 8000

# $PORT is provided by Render/Railway; defaults to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
