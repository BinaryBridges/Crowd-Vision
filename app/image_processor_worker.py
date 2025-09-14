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
    print(
        json.dumps({"event": "info", "detail": "Stream processing disabled - worker completed sample processing"}),
        flush=True,
    )

    # TODO: Enable this later when you want stream processing
    # try:
    #     _run_stream_loop(model)
    # except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, redis.exceptions.RedisError) as e:
    #     # Redis connection issues - k8s will restart the pod
    #     print(json.dumps({"event": "error", "detail": f"redis_error: {e}"}), flush=True)
    # except KeyboardInterrupt:
    #     # Graceful shutdown
    #     print(json.dumps({"event": "info", "detail": "worker_shutdown"}), flush=True)
    # except OSError as e:
    #     # Network or system-level issues
    #     print(json.dumps({"event": "error", "detail": f"system_error: {e}"}), flush=True)
