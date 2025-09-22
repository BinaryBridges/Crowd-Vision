import json
import os
import warnings

import numpy as np
import redis

# Force CPU for insightface/onnxruntime
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
from app.utils import format_demographics, reduce_embedding_for_tracking


# ---- model init ----
def _init_model() -> FaceAnalysis:
    app = FaceAnalysis(
        name=config.DETECTION_MODEL_NAME,
        allowed_modules=["detection", "genderage", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)
    return app


# ---- stable id helper (kept local to avoid touching utils.py) ----
def _stable_embedding_id(embedding: np.ndarray, decimals: int = 3) -> str:
    """
    Jitter-tolerant ID for a face embedding:
    1) reduce + L2 normalize (via existing reducer)
    2) round to 'decimals'
    3) sha256 hex
    """
    import hashlib

    red = reduce_embedding_for_tracking(np.asarray(embedding, dtype=np.float32))
    q = np.round(red.astype(np.float32), decimals=decimals)
    return hashlib.sha256(q.tobytes()).hexdigest()


# ---- core store w/ similarity dedupe ----
def _store_face_data(face, camera: str, frame: int | None = None, t_ns: int | None = None) -> None:
    try:
        embedding = getattr(face, "embedding", None)
        if embedding is None or len(embedding) == 0:
            print(
                json.dumps({
                    "event": "warning",
                    "detail": "No embedding found for face",
                    "camera": camera,
                    "frame": frame,
                }),
                flush=True,
            )
            return

        # reduced, L2-normalized vector for similarity
        red = reduce_embedding_for_tracking(np.asarray(embedding, dtype=np.float32))

        # similarity-first dedupe over recent faces
        dup_id = redis_client.find_similar_face(
            red,
            threshold=config.DEDUPE_SIM_THRESHOLD,
            recent=config.RECENT_FACE_WINDOW,
        )

        demographics = format_demographics(face)
        age_str = str(demographics.get("age")) if demographics.get("age") is not None else None
        gender_str = demographics.get("gender")

        if dup_id:
            redis_client.bump_existing_face(dup_id)
            print(
                json.dumps({
                    "event": "duplicate_face_skipped",
                    "embedding_id": dup_id,
                    "camera": camera,
                    "frame": frame,  # <-- NEW
                    "t_ns": t_ns,  # <-- NEW
                    "reason": "cosine>=0.92",
                }),
                flush=True,
            )
            return

        # new identity → compute stable ID and attempt atomic insert
        embedding_id = _stable_embedding_id(embedding, decimals=3)
        inserted = redis_client.save_face_if_new(
            embedding_id=embedding_id,
            age=age_str,
            gender=gender_str,
            cameras=camera,
        )

        if inserted:
            redis_client.record_new_face(
                embedding_id=embedding_id,
                vec=red,
                age=age_str,
                gender=gender_str,
                cameras=camera,
            )
            print(
                json.dumps({
                    "event": "face_stored_successfully",
                    "embedding_id": embedding_id,
                    "camera": camera,
                    "frame": frame,  # <-- NEW
                    "t_ns": t_ns,  # <-- NEW
                    "demographics": {"age": age_str, "gender": gender_str},
                }),
                flush=True,
            )
        else:
            redis_client.bump_existing_face(embedding_id)
            print(
                json.dumps({
                    "event": "duplicate_face_skipped",
                    "embedding_id": embedding_id,
                    "camera": camera,
                    "frame": frame,  # <-- NEW
                    "t_ns": t_ns,  # <-- NEW
                    "reason": "id_guard",
                }),
                flush=True,
            )

    except (AttributeError, TypeError, ValueError) as e:
        print(json.dumps({"event": "error", "detail": f"Face data processing error: {e}"}), flush=True)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, redis.exceptions.RedisError) as e:
        print(json.dumps({"event": "error", "detail": f"Redis storage error: {e}"}), flush=True)
    except OSError as e:
        print(json.dumps({"event": "error", "detail": f"System error during storage: {e}"}), flush=True)


def _process_image_array(
    model: FaceAnalysis, img_bgr, camera: str = "sample", *, frame: int | None = None, t_ns: int | None = None
) -> None:
    faces = model.get(img_bgr)
    # lightweight trace so you always see that the consumer handled a frame
    print(
        json.dumps({
            "event": "stream_msg_processed",
            "camera": camera,
            "frame": frame,
            "t_ns": t_ns,
            "faces_found": len(faces),
        }),
        flush=True,
    )
    for face in faces:
        _store_face_data(face, camera, frame=frame, t_ns=t_ns)


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
            frame = msg.get("frame")
            t_ns = msg.get("t_ns")

            # log receipt so you can correlate with the producer's "queued_message_id"
            print(
                json.dumps({
                    "event": "stream_msg_received",
                    "message_id": mid,
                    "camera": cam,
                    "frame": frame,
                    "t_ns": t_ns,
                }),
                flush=True,
            )

            try:
                if img is not None:
                    _process_image_array(model, img, cam, frame=frame, t_ns=t_ns)
                if mid:
                    redis_client.acknowledge_message(group, mid)
            except Exception as e:
                # don’t crash the whole worker on a single bad frame
                print(
                    json.dumps({
                        "event": "error",
                        "detail": f"image_processing_failed: {e}",
                        "message_id": mid,
                        "camera": cam,
                        "frame": frame,
                    }),
                    flush=True,
                )


def start_image_processor_worker():
    """
    Worker entrypoint: continuously consume frames from stream,
    run demographics, and write de-duplicated faces to Redis.
    """
    model = _init_model()
    try:
        _run_stream_loop(model)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, redis.exceptions.RedisError) as e:
        print(json.dumps({"event": "error", "detail": f"redis_error: {e}"}), flush=True)
        raise
    except KeyboardInterrupt:
        print(json.dumps({"event": "info", "detail": "worker_shutdown"}), flush=True)
    except OSError as e:
        print(json.dumps({"event": "error", "detail": f"system_error: {e}"}), flush=True)
