FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=http://ollama:11434 \
    GEMMA_FAST_MODEL=gemma4:e2b \
    FAST_USE_MINILM=0 \
    FAST_SKIP_GEMMA_ON_UPLOAD=1 \
    FAST_BACKEND_PORT=8011

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8011
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
