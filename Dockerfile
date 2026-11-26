FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-api.txt requirements-vector.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt

COPY src/ ./src/
COPY data/processed/fastapi_frozen_v2.jsonl ./data/processed/fastapi_frozen_v2.jsonl

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1

CMD ["python", "-m", "src.api_server", "--host", "0.0.0.0", "--port", "8000"]
