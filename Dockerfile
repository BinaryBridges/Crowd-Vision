FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies needed for OpenCV and insightface
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libgtk-3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code
COPY app ./app
COPY samples ./samples

# Pre-download InsightFace model during build to avoid startup delays
RUN python -c "import os; os.environ['INSIGHTFACE_ONNX_PROVIDERS'] = 'CPUExecutionProvider'; from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'genderage'], providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640, 640)); print('Model downloaded successfully')"

# Default entrypoint: python -m app
CMD ["python", "-m", "app"]