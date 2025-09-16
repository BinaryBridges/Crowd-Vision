FROM python:3.11

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code
COPY app ./app

# Default entrypoint: python -m app
CMD ["python", "-m", "app"]
