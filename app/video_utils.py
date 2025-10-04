"""
Video processing utilities for motion detection workers.
All utilities related to video file processing, path normalization, and Redis storage.
"""

import os
import pathlib

import redis

from app.redis_client import redis_client


def normalize_video_path(video_path: str) -> str:
    """
    Normalize video path for use in Redis keys by replacing path separators.
    This ensures consistent key generation across different operating systems.
    """
    return video_path.replace("/", ":").replace("\\", ":")


def get_video_lock_key(video_path: str) -> str:
    """Generate Redis key for video processing lock."""
    normalized_path = normalize_video_path(video_path)
    return f"video_lock:{normalized_path}"


def get_video_processed_key(video_path: str) -> str:
    """Generate Redis key for video processing completion marker."""
    normalized_path = normalize_video_path(video_path)
    return f"video_processed:{normalized_path}"


def get_camera_name_from_video_path(video_path: str) -> str:
    """
    Extract camera name from video file path.
    Uses the filename without extension as the camera identifier.
    """
    return pathlib.Path(video_path).stem


def get_video_storage_connection():
    """
    Get Redis connection for video processing metadata.
    Uses a separate database from face storage to avoid conflicts.
    """
    # Get video-specific database number (default to DB 2)
    video_db = int(os.getenv("REDIS_DB_VIDEO", "2"))

    # Create connection with same config as main storage but different DB
    video_storage = redis.Redis(
        host=redis_client.host,
        port=redis_client.port,
        db=video_db,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        decode_responses=True,  # For easier string handling
    )

    return video_storage
