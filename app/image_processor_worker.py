import json
import os
import warnings

import redis

os.environ["INSIGHTFACE_ONNX_PROVIDERS"] = "CPUExecutionProvider"
os.environ["ORT_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings(
    "ignore",
    message="Specified provider 'CUDAExecutionProvider' is not in available provider names",
    category=UserWarning,
)

import cv2

cv2.setNumThreads(1)

from insightface.app import FaceAnalysis

from app import config
from app.redis_client import redis_client
from app.utils import format_demographics, load_bgr_image


def _init_model() -> FaceAnalysis:
    app = FaceAnalysis(
        name=config.DETECTION_MODEL_NAME,
        allowed_modules=["detection", "genderage"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)
    return app


def _log_demographics(camera: str, faces) -> None:
    payload = {
        "event": "demographics",
        "camera": camera,
        "count": len(faces),
        "faces": [format_demographics(f) for f in faces],
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _process_image_array(model: FaceAnalysis, img_bgr, camera: str = "sample") -> None:
    faces = model.get(img_bgr)
    _log_demographics(camera, faces)


def _run_once_on_sample(model: FaceAnalysis) -> None:
    img = load_bgr_image(config.SAMPLE_IMAGE)
    _process_image_array(model, img, camera="sample")


def _run_stream_loop(model: FaceAnalysis) -> None:
    group = config.REDIS_CONSUMER_GROUP
    name = config.REDIS_CONSUMER_NAME

    while True:
        msgs = redis_client.take_image_from_stream(
            consumer_group=group,
            consumer_name=name,
            count=1,
            block=5000,
        )
        if not msgs:
            continue

        for msg in msgs:
            img = msg.get("image")
            cam = msg.get("camera", "unknown")
            mid = msg.get("message_id")

            if img is not None:
                _process_image_array(model, img, cam)

            redis_client.acknowledge_message(group, mid)


def start_image_processor_worker():
    """
    Worker entrypoint:
    - For now: process the sample image once and log demographics.
    - Later (k8s stream): remove the sample step and rely solely on the stream loop.
    """
    model = _init_model()
    _run_once_on_sample(model)
    try:
        _run_stream_loop(model)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, redis.exceptions.RedisError) as e:
        print(json.dumps({"event": "error", "detail": f"redis_error: {e}"}), flush=True)
    except KeyboardInterrupt:
        print(json.dumps({"event": "info", "detail": "worker_shutdown"}), flush=True)
    except OSError as e:
        print(json.dumps({"event": "error", "detail": f"system_error: {e}"}), flush=True)
