import logging
import os
import pathlib
import time

import cv2
import redis

from app.performance_utils import PerformanceMonitor, VideoProcessor, optimize_opencv_performance, timer
from app.redis_client import redis_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


FORCE_ENQUEUE_FIRST_FRAME = True  # can move to config if you prefer
ENQUEUE_EVERY_N_FRAMES = 0  # 0 = disabled; set e.g. 30 to sample


def _process_single_frame(image, last_image_resized, performance_monitor, *, frame_index: int):
    resized_current = redis_client.resize_for_motion_detection(image)

    queued_mid = None
    motion_detected = False
    reason = None

    # 1) Bootstrap: send first frame no matter what
    if frame_index == 1 and FORCE_ENQUEUE_FIRST_FRAME:
        queued_mid = redis_client.put_image_in_stream(
            image,
            "camera_001",
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
                "camera_001",
                frame=frame_index,
                t_ns=time.time_ns(),
            )
            reason = "periodic"

        # Enqueue on motion
        if not queued_mid and motion_detected:
            queued_mid = redis_client.put_image_in_stream(
                image,
                "camera_001",
                frame=frame_index,
                t_ns=time.time_ns(),
            )
            reason = "motion"

    # Per-frame log (now includes reason if enqueued)
    logger.info(
        "Frame %d: motion=%s%s%s",
        frame_index,
        motion_detected,
        f" enqueued_reason={reason}" if reason else "",
        f" queued_message_id={queued_mid}" if queued_mid else "",
    )

    performance_monitor.update()
    return resized_current


def _process_video_frames(path_in):
    """Process video frames for motion detection."""
    with VideoProcessor(path_in, target_fps=1.0) as video_proc:
        performance_monitor = PerformanceMonitor("MotionDetection")
        last_image_resized = None

        logger.info("Starting simple frame processing")

        for processed_count, (success, image) in enumerate(video_proc.read_frames(), start=1):
            if not success:
                break

            last_image_resized = _process_single_frame(
                image,
                last_image_resized,
                performance_monitor,
                frame_index=processed_count,  # <-- NEW
            )

            if processed_count % 10 == 0:
                performance_monitor.log_stats()

        final_stats = performance_monitor.get_stats()
        logger.info("=== PROCESSING COMPLETE ===")
        logger.info("Total frames processed: %s", final_stats["frames_processed"])
        logger.info("Total processing time: %.2f seconds", final_stats["elapsed_time"])
        logger.info("Average processing speed: %.2f fps", final_stats["average_fps"])


def start_motion_detection_worker():
    """Optimized main loop for motion detection worker."""
    logger.info("Starting optimized motion detection worker")

    # Test Redis connection before starting
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            redis_client.ping()
            logger.info("Redis connection successful")
            break
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            logger.warning("Redis connection attempt %s / %s failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                logger.info("Retrying in %s seconds...", retry_delay)
                time.sleep(retry_delay)
            else:
                logger.exception("Failed to connect to Redis after all retries")
                return 1

    # Apply OpenCV optimizations
    optimize_opencv_performance()

    path_in = "./app/video/people_walking_1080.mp4"

    # Check if video file exists
    if not pathlib.Path(path_in).exists():
        logger.error("Video file does not exist: %s", path_in)
        logger.info("Current working directory: %s", pathlib.Path.cwd())
        logger.info("Files in current directory: %s", os.listdir("."))
        return 0

    # Use optimized video processor
    try:
        _process_video_frames(path_in)

    except (
        redis.exceptions.ConnectionError,
        redis.exceptions.TimeoutError,
        redis.exceptions.RedisError,
        cv2.error,
        OSError,
    ):
        # Operational errors we expect and want to report/exit on
        logger.exception("Error in motion detection worker")
        # If you have cleanup to do, do it here before returning
        raise  # or `return 1` if this is inside a function and you prefer exit codes

    except KeyboardInterrupt:
        logger.info("Motion detection worker interrupted by user")
        raise  # or `return 0` if you intentionally treat this as a clean stop

    # If not in a function, you can continue; if in a function, return success code
    logger.info("Motion detection worker completed successfully")
    return 0


def process_task(task=None):
    """Process a single unit of work."""


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
