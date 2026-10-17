# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# rag-doc-qa image.
#
# Two runtime modes, both supported by this single image:
#   1. Point OPENAI_BASE_URL at an already-running gateway (Ollama, vLLM,
#      llama.cpp server, a hosted API). This is the lean, recommended path.
#   2. Run the `api` service in docker-compose.yml, which also starts an Ollama
#      sidecar. Heavier, but gives a self-contained demo.
#
# The image installs fastembed + onnxruntime so the real embedding and
# cross-encoder reranking path is active. Models are downloaded on first ingest
# and cached in the `models` volume so restarts don't re-download.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS base

# sqlite-vec ships a prebuilt .so loaded via SQLite's extension mechanism; it
# needs libstdc++ at runtime. pypdf and fastembed are pure-python/ONNX.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# FastEmbed/ONNX model cache location (also mounted as a volume in compose).
ENV HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

COPY pyproject.toml README.md ./
COPY src ./src
COPY cli.py conftest.py ./
COPY data/corpus ./data/corpus
COPY data/eval ./data/eval

RUN pip install -e . --no-deps

# Non-root user; the index and model caches are writable volumes.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default: serve the API. `docker compose run cli ...` overrides this to run the
# CLI (ingest/ask/eval) against the same volumes.
CMD ["uvicorn", "ragqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
