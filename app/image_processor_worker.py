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
from app.utils import format_demographics


def _init_model() -> FaceAnalysis:
    app = FaceAnalysis(
        name=config.DETECTION_MODEL_NAME,
        allowed_modules=["detection", "genderage", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)
    return app


def _store_face_data(face, camera: str) -> None:
    """Store face data in Redis database."""
    try:
        # Get the embedding from the face object
        embedding = getattr(face, "embedding", None)
        if embedding is None or len(embedding) == 0:
            print(json.dumps({"event": "warning", "detail": "No embedding found for face"}), flush=True)
            return

        embedding_str = str(hash(tuple(embedding.flatten())))  # Simple hash for now

        # Get demographics
        demographics = format_demographics(face)
        age_str = str(demographics.get("age")) if demographics.get("age") is not None else None
        gender_str = demographics.get("gender")

        # Log what we're about to store
        print(
            json.dumps({
                "event": "storing_face_data",
                "embedding_id": embedding_str,
                "age": age_str,
                "gender": gender_str,
                "camera": camera,
                "embedding_length": len(embedding),
            }),
            flush=True,
        )

        # Store in Redis
        redis_client.save_face_data(embedding=embedding_str, age=age_str, gender=gender_str, cameras=camera)

        print(
            json.dumps({
                "event": "face_stored_successfully",
                "embedding_id": embedding_str,
                "camera": camera,
                "demographics": {"age": age_str, "gender": gender_str},
            }),
            flush=True,
        )

    except (AttributeError, TypeError, ValueError) as e:
        # Face object issues, embedding conversion issues, or data type issues
        print(json.dumps({"event": "error", "detail": f"Face data processing error: {e}"}), flush=True)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, redis.exceptions.RedisError) as e:
        # Redis-specific errors
        print(json.dumps({"event": "error", "detail": f"Redis storage error: {e}"}), flush=True)
    except OSError as e:
        # Network or system-level issues
        print(json.dumps({"event": "error", "detail": f"System error during storage: {e}"}), flush=True)


def _process_image_array(model: FaceAnalysis, img_bgr, camera: str = "sample") -> None:
    faces = model.get(img_bgr)

    for face in faces:
        _store_face_data(face, camera)


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
    model = _init_model()

    try:
        _run_stream_loop(model)
    except KeyboardInterrupt:
        print(json.dumps({"event": "info", "detail": "worker_shutdown"}), flush=True)
    except Exception as e:
        print(json.dumps({"event": "error", "detail": f"worker_crashed: {e}"}), flush=True)
        raise
