import json
import os
import socket
import threading
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
from app.face_utils import (
    format_demographics,
    reduce_embedding_for_tracking,
    save_face_if_new,
    stable_embedding_id,
)
from app.redis_client import redis_client


# ---- model init ----
def _init_model() -> FaceAnalysis:
    app = FaceAnalysis(
        name=config.DETECTION_MODEL_NAME,
        allowed_modules=["detection", "genderage", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)
    return app


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

        # similarity-first dedupe using RedisSearch
        dup_id = redis_client.find_similar_face_redisearch(
            red,
            threshold=config.DEDUPE_SIM_THRESHOLD,
            limit=50,  # Check top 50 most similar faces
        )

        demographics = format_demographics(face)
        age_val = demographics.get("age")
        age_str = str(age_val) if age_val is not None else None
        gender_str = demographics.get("gender")

        if dup_id:
            # Bump the seen count and get the new value
            new_seen_count = redis_client.bump_existing_face(dup_id)

            # Check if we should update with moving average
            if new_seen_count % config.FACE_UPDATE_INTERVAL == 0:
                # Time to recalculate - update with moving average
                redis_client.update_face_with_moving_average(
                    embedding_id=dup_id,
                    new_vec=red,
                    new_age=age_val,
                    seen_count=new_seen_count,
                )
                print(
                    json.dumps({
                        "event": "face_updated_moving_average",
                        "embedding_id": dup_id,
                        "camera": camera,
                        "frame": frame,
                        "t_ns": t_ns,
                        "seen_count": new_seen_count,
                        "demographics": {"age": age_str, "gender": gender_str},
                    }),
                    flush=True,
                )
            else:
                print(
                    json.dumps({
                        "event": "duplicate_face_skipped",
                        "embedding_id": dup_id,
                        "camera": camera,
                        "frame": frame,
                        "t_ns": t_ns,
                        "seen_count": new_seen_count,
                        "reason": "cosine>=0.92",
                    }),
                    flush=True,
                )
            return

        # new identity → compute stable ID and attempt atomic insert using separated storage
        embedding_id = stable_embedding_id(embedding, decimals=3)
        inserted = save_face_if_new(
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
                    "frame": frame,
                    "t_ns": t_ns,
                    "demographics": {"age": age_str, "gender": gender_str},
                }),
                flush=True,
            )
        else:
            # Bump the seen count and get the new value
            new_seen_count = redis_client.bump_existing_face(embedding_id)

            # Check if we should update with moving average
            if new_seen_count % config.FACE_UPDATE_INTERVAL == 0:
                # Time to recalculate - update with moving average
                redis_client.update_face_with_moving_average(
                    embedding_id=embedding_id,
                    new_vec=red,
                    new_age=age_val,
                    seen_count=new_seen_count,
                )
                print(
                    json.dumps({
                        "event": "face_updated_moving_average",
                        "embedding_id": embedding_id,
                        "camera": camera,
                        "frame": frame,
                        "t_ns": t_ns,
                        "seen_count": new_seen_count,
                        "demographics": {"age": age_str, "gender": gender_str},
                    }),
                    flush=True,
                )
            else:
                print(
                    json.dumps({
                        "event": "duplicate_face_skipped",
                        "embedding_id": embedding_id,
                        "camera": camera,
                        "frame": frame,
                        "t_ns": t_ns,
                        "seen_count": new_seen_count,
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
                    # Acknowledge and delete the message to remove it from the stream
                    redis_client.acknowledge_and_delete_message(group, mid)
            except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
                # don't crash the whole worker on a single bad frame
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


def _run_stream_loop_with_heartbeat(model: FaceAnalysis, worker_id: str) -> None:
    """Stream processing loop with heartbeat status updates."""
    group = config.REDIS_CONSUMER_GROUP
    name = config.REDIS_CONSUMER_NAME

    while True:
        # Update heartbeat status to indicate waiting for work
        redis_client.update_worker_heartbeat(worker_id, "waiting_for_messages")

        msgs = redis_client.take_image_from_stream(
            consumer_group=group,
            consumer_name=name,
            count=1,
            block=5000,
        )
        if not msgs:
            continue

        # Update heartbeat status to indicate processing
        redis_client.update_worker_heartbeat(worker_id, "processing")

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
                    # Acknowledge and delete the message to remove it from the stream
                    redis_client.acknowledge_and_delete_message(group, mid)
                    print(
                        json.dumps({
                            "event": "stream_msg_completed",
                            "message_id": mid,
                            "camera": cam,
                            "frame": frame,
                            "detail": "message_deleted_from_stream",
                        }),
                        flush=True,
                    )
            except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
                # don't crash the whole worker on a single bad frame
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
                # Still acknowledge and delete the message even if processing failed
                # to prevent it from being reprocessed indefinitely
                if mid:
                    try:
                        redis_client.acknowledge_and_delete_message(group, mid)
                        print(
                            json.dumps({
                                "event": "stream_msg_completed",
                                "message_id": mid,
                                "detail": "failed_message_deleted_from_stream",
                            }),
                            flush=True,
                        )
                    except redis.exceptions.RedisError as cleanup_error:
                        print(
                            json.dumps({
                                "event": "error",
                                "detail": f"failed_to_cleanup_message: {cleanup_error}",
                                "message_id": mid,
                            }),
                            flush=True,
                        )


def start_image_processor_worker():
    """
    Worker entrypoint: continuously consume frames from stream,
    run demographics, and write de-duplicated faces to Redis.
    """
    print(json.dumps({"event": "info", "detail": "Starting image processor worker"}), flush=True)

    # Wait for Redis to be ready with comprehensive health checks
    if not redis_client.wait_for_redis(timeout=300):  # 5 minute timeout
        print(json.dumps({"event": "error", "detail": "Redis failed to become ready. Exiting."}), flush=True)
        return 1

    # Generate unique worker ID and register with Redis
    worker_id = f"image-{socket.gethostname()[:15]}-{os.getpid()}"
    redis_client.register_worker(worker_id, "image-processor")

    # Set up heartbeat thread
    heartbeat_stop = threading.Event()

    def heartbeat_loop():
        while not heartbeat_stop.is_set():
            redis_client.update_worker_heartbeat(worker_id, "active")
            heartbeat_stop.wait(30)  # Update every 30 seconds

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    print(json.dumps({"event": "info", "detail": "Redis is ready, initializing face analysis model"}), flush=True)
    model = _init_model()

    print(json.dumps({"event": "info", "detail": "Starting stream processing loop"}), flush=True)
    try:
        _run_stream_loop_with_heartbeat(model, worker_id)
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, redis.exceptions.RedisError) as e:
        print(json.dumps({"event": "error", "detail": f"redis_error: {e}"}), flush=True)
        heartbeat_stop.set()
        redis_client.unregister_worker(worker_id)
        raise
    except KeyboardInterrupt:
        print(json.dumps({"event": "info", "detail": "worker_shutdown"}), flush=True)
        heartbeat_stop.set()
        redis_client.unregister_worker(worker_id)
    except OSError as e:
        print(json.dumps({"event": "error", "detail": f"system_error: {e}"}), flush=True)
        heartbeat_stop.set()
        redis_client.unregister_worker(worker_id)
        raise
    finally:
        # Stop heartbeat thread
        heartbeat_stop.set()
