"""
Face metadata maintenance utilities.
Utilities for cleaning up and monitoring face metadata storage.
"""

import redis

from app.face_utils import get_face_guard_key, get_face_metadata_storage_connection, get_face_similarity_key


def cleanup_face_metadata(embedding_id: str) -> None:
    """
    Clean up face metadata (guards, similarity data) without affecting face data.
    Useful for maintenance and testing.
    """
    metadata_storage = get_face_metadata_storage_connection()

    guard_key = get_face_guard_key(embedding_id)
    similarity_key = get_face_similarity_key(embedding_id)

    # Clean up metadata without touching face embeddings/demographics
    metadata_storage.delete(guard_key, similarity_key)


def get_face_metadata_stats() -> dict:
    """
    Get statistics about face metadata storage.
    Useful for monitoring and maintenance.
    """
    metadata_storage = get_face_metadata_storage_connection()

    try:
        info = metadata_storage.info()
        return {
            "db_size": info.get("db0", {}).get("keys", 0),
            "memory_usage": info.get("used_memory_human", "unknown"),
            "guard_keys": len(list(metadata_storage.scan_iter(match="face:*:created"))),
            "similarity_keys": len(list(metadata_storage.scan_iter(match="face_sim:*"))),
        }
    except redis.exceptions.RedisError as e:
        return {"error": str(e)}
