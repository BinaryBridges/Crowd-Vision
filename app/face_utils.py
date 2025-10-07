"""
Face processing utilities for image processor workers.
All utilities related to face embeddings, demographics, and Redis storage.
"""

import contextlib
import hashlib
import os
from typing import Any

import numpy as np
import redis

from app import config
from app.redis_client import redis_client


def _l2(v: np.ndarray) -> np.ndarray:
    """L2 normalization for embeddings."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def reduce_embedding_for_tracking(emb: np.ndarray) -> np.ndarray:
    """
    Reduce embedding dimensionality to VECTOR_SIZE and L2 normalize.
    Used for consistent face tracking across the system.
    """
    emb = np.asarray(emb, dtype=np.float32)
    k = min(config.VECTOR_SIZE, emb.shape[-1])
    return _l2(emb[..., :k])


def _short_gender(g) -> str | None:
    """Convert gender representation to 'M' or 'F' or None."""
    if isinstance(g, (int, np.integer)):
        return "M" if int(g) == 1 else "F"
    if isinstance(g, str):
        s = g.lower()
        if s.startswith("m"):
            return "M"
        if s.startswith("f"):
            return "F"
    return None


def format_demographics(face_obj) -> dict[str, Any]:
    """
    Extract and format demographics from a face object.
    Returns {'gender': 'M'|'F'|None, 'age': int|None}.
    """
    g = getattr(face_obj, "gender", None)
    if g is None:
        g = getattr(face_obj, "sex", None)
    gender = _short_gender(g)

    age_val = getattr(face_obj, "age", None)
    age_out = None
    if age_val is not None:
        with contextlib.suppress(Exception):
            age_out = round(float(age_val))

    return {"gender": gender, "age": age_out}


def stable_embedding_id(embedding: np.ndarray, decimals: int = 3) -> str:
    """
    Produce a stable, jitter-tolerant ID for a face embedding.
    1) reduce + L2 normalize
    2) round (quantize) to 'decimals'
    3) sha256 hash to hex string
    """
    red = reduce_embedding_for_tracking(np.asarray(embedding))
    q = np.round(red.astype(np.float32), decimals=decimals)
    b = q.tobytes()
    return hashlib.sha256(b).hexdigest()


def get_face_key(embedding_id: str) -> str:
    """Generate Redis key for face data (embeddings, demographics, vectors) in DB 0."""
    return embedding_id  # Clean key since DB 0 is exclusive to face data


def get_face_guard_key(embedding_id: str) -> str:
    """Generate Redis key for face deduplication guard in DB 3."""
    return embedding_id  # Clean key since DB 3 is exclusive to deduplication guards


def get_face_similarity_key(embedding_id: str) -> str:
    """Generate Redis key for face similarity metadata in DB 3."""
    return f"sim:{embedding_id}"  # Minimal prefix to distinguish from guards


def get_face_metadata_storage_connection():
    """
    Get Redis connection for face deduplication metadata and guards.
    Uses a separate database from face embeddings to optimize search performance.
    """
    # Get face metadata database number (default to DB 3)
    face_metadata_db = int(os.getenv("REDIS_DB_FACE_METADATA", "3"))

    # Create connection with same config as main storage but different DB
    face_metadata_storage = redis.Redis(
        host=redis_client.host,
        port=redis_client.port,
        db=face_metadata_db,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        decode_responses=True,  # For easier string handling
    )

    return face_metadata_storage


def save_face_if_new(*, embedding_id: str, age: str | None, gender: str | None, cameras: str) -> bool:
    """
    Atomic id-based guard using separated storage for optimal performance.

    Face data goes to DB 0 (for search optimization)
    Deduplication guard goes to DB 3 (for cleanup safety)

    Returns True if this process created the entry, False if existed.
    """
    face_key = get_face_key(embedding_id)
    guard_key = get_face_guard_key(embedding_id)

    # Get separate storage connections
    face_storage = redis_client.storage  # DB 0 - for search performance
    metadata_storage = get_face_metadata_storage_connection()  # DB 3 - for guards

    # Try to set the guard atomically in the metadata database
    if metadata_storage.setnx(guard_key, "1"):
        # Guard acquired, now store the face data in the main database
        mapping = {"confidence": "", "age": age or "", "gender": gender or "", "cameras": cameras}
        face_storage.hset(face_key, mapping=mapping)
        return True

    return False
