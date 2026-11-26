# ================= Target: trainer =================

FROM python:3.11-slim AS trainer

WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt requirements-train.txt ./

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements-train.txt

COPY src/ ./src/

COPY ci_dataset/ ./ci_dataset/

ENTRYPOINT ["python", "-m", "src.train"]

CMD ["--classes", "yes", "no", "up", "down", \
     "--epochs", "10", \
     "--seed", "42", \
     "--dataset", "ci_dataset", \
     "--out", "/artifacts"]


# ================= Target: runtime =================

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements-api.txt

COPY src/ ./src/

COPY artifacts/ ./artifacts/

ENV MODEL_PATH=/app/artifacts/model.pt \
    LABELS_PATH=/app/artifacts/labels.json \
    METRICS_PATH=/app/artifacts/metrics.json \
    MODEL_VERSION=1.3.0 \
    PORT=8000 \
    LOG_LEVEL=info

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT} --log-level ${LOG_LEVEL}"]