import glob
import logging
import os
import pathlib
import socket
import threading
import time

import cv2
import redis

from app.performance_utils import PerformanceMonitor, VideoProcessor, optimize_opencv_performance, timer
from app.redis_client import redis_client
from app.video_utils import (
    get_camera_name_from_video_path,
    get_video_lock_key,
    get_video_processed_key,
    get_video_storage_connection,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


FORCE_ENQUEUE_FIRST_FRAME = True  # can move to config if you prefer
ENQUEUE_EVERY_N_FRAMES = 0  # 0 = disabled; set e.g. 30 to sample

# Video processing configuration
VIDEO_LOAD_DIR = "./app/video/load"
VIDEO_EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.webm"]


def discover_videos() -> list[str]:
    """Discover all video files in the load directory."""
    video_files = []

    if not pathlib.Path(VIDEO_LOAD_DIR).exists():
        logger.warning("Video load directory does not exist: %s", VIDEO_LOAD_DIR)
        return video_files

    for extension in VIDEO_EXTENSIONS:
        pattern = os.path.join(VIDEO_LOAD_DIR, extension)
        video_files.extend(glob.glob(pattern))

    # Convert to relative paths and sort for consistent processing order
    video_files = [os.path.relpath(path) for path in video_files]
    video_files.sort()

    logger.info("Discovered %d video files: %s", len(video_files), video_files)
    return video_files


def try_lock_video(video_path: str) -> bool:
    """
    Try to acquire a permanent lock for processing a video file.
    Returns True if lock acquired, False if already locked.
    The lock never expires to ensure videos are never processed twice.
    Uses separate Redis database for video metadata.
    """
    lock_key = get_video_lock_key(video_path)
    video_storage = get_video_storage_connection()

    try:
        # Use SETNX to create an atomic lock that never expires
        # This will set the key only if it doesn't exist
        result = video_storage.set(
            lock_key,
            f"locked_by_worker_at_{time.time()}",
            nx=True,  # Only set if not exists - no expiry
        )

        if result:
            logger.info("Successfully acquired permanent lock for video: %s", video_path)
            return True
        else:
            logger.info("Video already locked by another worker: %s", video_path)
            return False

    except redis.exceptions.RedisError:
        logger.exception("Failed to acquire lock for video %s", video_path)
        return False


def release_video_lock(video_path: str) -> None:
    """
    Release the permanent lock for a video file.
    This should only be called after successful processing completion.
    Uses separate Redis database for video metadata.
    """
    lock_key = get_video_lock_key(video_path)
    video_storage = get_video_storage_connection()

    try:
        video_storage.delete(lock_key)
        logger.info("Released permanent lock for video: %s", video_path)
    except redis.exceptions.RedisError:
        logger.exception("Failed to release lock for video %s", video_path)


def mark_video_processed(video_path: str) -> None:
    """
    Mark a video as completely processed.
    Uses separate Redis database for video metadata.
    """
    processed_key = get_video_processed_key(video_path)
    video_storage = get_video_storage_connection()

    try:
        # Store processing completion timestamp
        video_storage.set(processed_key, int(time.time()))
        logger.info("Marked video as processed: %s", video_path)
    except redis.exceptions.RedisError:
        logger.exception("Failed to mark video as processed %s", video_path)


def is_video_processed(video_path: str) -> bool:
    """
    Check if a video has already been processed.
    Uses separate Redis database for video metadata.
    """
    processed_key = get_video_processed_key(video_path)
    video_storage = get_video_storage_connection()

    try:
        return video_storage.exists(processed_key)
    except redis.exceptions.RedisError:
        logger.exception("Failed to check if video is processed %s", video_path)
        return False


def get_next_video_to_process() -> str | None:
    """
    Find the next video that needs processing.
    Returns the video path if found, None if all videos are processed or locked.
    """
    videos = discover_videos()

    for video_path in videos:
        # Skip if already processed
        if is_video_processed(video_path):
            logger.debug("Video already processed, skipping: %s", video_path)
            continue

        # Try to acquire lock
        if try_lock_video(video_path):
            return video_path

    return None


def list_video_status() -> dict:
    """
    Get the status of all videos in the load directory.
    Returns a dictionary with video paths and their status.
    Useful for debugging and monitoring.
    Uses separate Redis database for video metadata.
    """
    videos = discover_videos()
    status = {}
    video_storage = get_video_storage_connection()

    for video_path in videos:
        lock_key = get_video_lock_key(video_path)
        processed_key = get_video_processed_key(video_path)

        try:
            is_locked = video_storage.exists(lock_key)
            is_processed = video_storage.exists(processed_key)

            if is_processed:
                status[video_path] = "completed"
            elif is_locked:
                status[video_path] = "locked"
            else:
                status[video_path] = "available"

        except redis.exceptions.RedisError:
            logger.exception("Failed to check status for video %s", video_path)
            status[video_path] = "unknown"

    return status


def _process_single_frame(image, last_image_resized, performance_monitor, *, frame_index: int, camera_name: str):
    resized_current = redis_client.resize_for_motion_detection(image)

    queued_mid = None
    motion_detected = False
    reason = None

    # 1) Bootstrap: send first frame no matter what
    if frame_index == 1 and FORCE_ENQUEUE_FIRST_FRAME:
        queued_mid = redis_client.put_image_in_stream(
            image,
            camera_name,
            frame=frame_index,
            t_ns=time.time_ns(),
        )
        reason = "first_frame"
    else:
        # 2) Normal motion path
        if last_image_resized is not None:
            with timer("motion_detection"):
                motion_detected = motion_detection(
                    last_image_resized, resized_current, threshold=30, min_motion_pixels=50
                )
        # 3) Periodic sampling if you want some non-motion frames too
        if not queued_mid and ENQUEUE_EVERY_N_FRAMES and frame_index % ENQUEUE_EVERY_N_FRAMES == 0:
            queued_mid = redis_client.put_image_in_stream(
                image,
                camera_name,
                frame=frame_index,
                t_ns=time.time_ns(),
            )
            reason = "periodic"

        # Enqueue on motion
        if not queued_mid and motion_detected:
            queued_mid = redis_client.put_image_in_stream(
                image,
                camera_name,
                frame=frame_index,
                t_ns=time.time_ns(),
            )
            reason = "motion"

    # Per-frame log (now includes reason if enqueued)
    logger.info(
        "[%s] Frame %d: motion=%s%s%s",
        camera_name,
        frame_index,
        motion_detected,
        f" enqueued_reason={reason}" if reason else "",
        f" queued_message_id={queued_mid}" if queued_mid else "",
    )

    performance_monitor.update()
    return resized_current


def _process_video_frames(path_in: str):
    """Process video frames for motion detection."""
    # Extract camera name from video file path for better identification
    camera_name = get_camera_name_from_video_path(path_in)

    logger.info("Starting video processing for: %s (camera: %s)", path_in, camera_name)

    with VideoProcessor(path_in, target_fps=1.0) as video_proc:
        performance_monitor = PerformanceMonitor("MotionDetection")
        last_image_resized = None

        logger.info("Starting frame processing for video: %s", path_in)

        for processed_count, (success, image) in enumerate(video_proc.read_frames(), start=1):
            if not success:
                break

            last_image_resized = _process_single_frame(
                image,
                last_image_resized,
                performance_monitor,
                frame_index=processed_count,
                camera_name=camera_name,  # Pass camera name to frame processor
            )

            if processed_count % 10 == 0:
                performance_monitor.log_stats()

        final_stats = performance_monitor.get_stats()
        logger.info("=== VIDEO PROCESSING COMPLETE ===")
        logger.info("Video: %s", path_in)
        logger.info("Total frames processed: %s", final_stats["frames_processed"])
        logger.info("Total processing time: %.2f seconds", final_stats["elapsed_time"])
        logger.info("Average processing speed: %.2f fps", final_stats["average_fps"])


def start_motion_detection_worker():
    """Main loop for motion detection worker that processes videos from load directory."""
    logger.info("Starting motion detection worker with video directory processing")

    # Wait for Redis to be ready with comprehensive health checks
    if not redis_client.wait_for_redis(timeout=300):  # 5 minute timeout
        logger.error("Redis failed to become ready. Exiting.")
        return 1

    logger.info("Redis is ready, starting motion detection worker")

    # Generate unique worker ID and register with Redis
    worker_id = f"motion-{socket.gethostname()[:15]}-{os.getpid()}"
    redis_client.register_worker(worker_id, "motion-detection")

    # Set up heartbeat thread
    heartbeat_stop = threading.Event()

    def heartbeat_loop():
        while not heartbeat_stop.is_set():
            redis_client.update_worker_heartbeat(worker_id, "active")
            heartbeat_stop.wait(30)  # Update every 30 seconds

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    # Apply OpenCV optimizations
    optimize_opencv_performance()

    # Check if video load directory exists
    if not pathlib.Path(VIDEO_LOAD_DIR).exists():
        logger.error("Video load directory does not exist: %s", VIDEO_LOAD_DIR)
        logger.info("Current working directory: %s", pathlib.Path.cwd())
        heartbeat_stop.set()
        return 1

    logger.info("Video load directory: %s", VIDEO_LOAD_DIR)

    # Main processing loop
    processed_count = 0

    try:
        while True:
            # Update heartbeat status
            redis_client.update_worker_heartbeat(worker_id, "searching_for_work")

            # Get next video to process
            video_path = get_next_video_to_process()

            if video_path is None:
                logger.info("No more videos to process. All videos completed or locked by other workers.")
                break

            logger.info("Processing video: %s", video_path)
            redis_client.update_worker_heartbeat(worker_id, "processing")

            try:
                # Check if video file exists and is accessible
                if not pathlib.Path(video_path).exists():
                    logger.error("Video file does not exist: %s", video_path)
                    # Don't release lock - keep it to prevent future attempts on missing file
                    logger.warning(
                        "Keeping lock for missing video to prevent future processing attempts: %s", video_path
                    )
                    continue

                # Process the video
                _process_video_frames(video_path)

                # Mark as processed and release lock only on successful completion
                mark_video_processed(video_path)
                release_video_lock(video_path)

                processed_count += 1
                logger.info("Successfully processed video %d: %s", processed_count, video_path)

            except (cv2.error, OSError):
                logger.exception("Error processing video %s", video_path)
                # Don't release lock on processing errors - keep it to prevent retries
                logger.warning("Keeping lock for failed video to prevent future processing attempts: %s", video_path)
                continue

            except (
                redis.exceptions.ConnectionError,
                redis.exceptions.TimeoutError,
                redis.exceptions.RedisError,
            ):
                logger.exception("Redis error while processing video %s", video_path)
                # Don't release lock on Redis errors - keep the permanent lock
                raise

    except KeyboardInterrupt:
        logger.info("Motion detection worker interrupted by user")
        # Try to release any current lock gracefully
        heartbeat_stop.set()
        redis_client.unregister_worker(worker_id)
        return 0

    except Exception:
        logger.exception("Unexpected error in motion detection worker")
        heartbeat_stop.set()
        redis_client.unregister_worker(worker_id)
        return 1
    finally:
        # Stop heartbeat thread
        heartbeat_stop.set()

    # Normal completion - unregister worker
    redis_client.unregister_worker(worker_id)
    logger.info("Motion detection worker completed successfully. Processed %d videos.", processed_count)
    return 0


def shutdown():
    """Optional: graceful shutdown."""


def motion_detection(image_1, image_2, threshold=25, min_motion_pixels=100):
    """
    Optimized motion detection between two consecutive frames.

    Args:
        image_1: First frame (background/reference frame) as BGR image
        image_2: Second frame (current frame) as BGR image
        threshold: Pixel difference threshold for motion detection
        min_motion_pixels: Minimum number of changed pixels to consider as motion

    Returns:
        bool: True if motion detected, False otherwise

    """
    # Step 1: Convert both images from BGR color to grayscale
    # Grayscale conversion reduces computational complexity and focuses on intensity changes
    g_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    g_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    # Step 2: Calculate absolute difference between the two grayscale frames
    # This highlights pixels that have changed between frames
    diff = cv2.absdiff(g_image_1, g_image_2)

    # Step 3: Apply threshold and check if any pixels exceed the motion threshold
    # Create binary image where pixels > threshold are white (255), others black (0)
    # This filters out minor variations due to noise or compression artifacts
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Step 4: Count motion pixels and compare against minimum threshold
    # This helps filter out minor noise and ensures significant motion
    motion_pixel_count = cv2.countNonZero(thresh)
    motion_detected = motion_pixel_count > min_motion_pixels

    return motion_detected
