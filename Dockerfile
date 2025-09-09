FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements (empty for now; keeps layer stable if you add later)
COPY requirements.txt .
RUN pip install -r requirements.txt || true

# App code
COPY app ./app

# Default entrypoint: python -m app
CMD ["python", "-m", "app"]
