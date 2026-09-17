FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN addgroup --system app \
    && adduser \
        --system \
        --ingroup app \
        app

COPY requirements/runtime.txt \
    /app/requirements/runtime.txt

RUN python -m pip install \
        --upgrade pip \
    && python -m pip install \
        -r /app/requirements/runtime.txt

COPY src/ /app/src/
COPY app/ /app/app/
COPY config/ /app/config/
COPY compose.yaml /app/compose.yaml

RUN mkdir -p /app/logs \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["python", "-c", "import uvicorn; from olist_ml.config import load_config; c = load_config()['service']; uvicorn.run('app.main:app', host=str(c['host']), port=int(c['port']))"]
