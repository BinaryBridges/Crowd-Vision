import sys
import argparse
import logging
import cv2
import numpy as np
import os
import time
from app.redis_client import redis_client
from app.performance_utils import VideoProcessor, PerformanceMonitor, timer, optimize_opencv_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_motion_detection_worker():
    """Optimized main loop for motion detection worker."""
    logger.info("Starting optimized motion detection worker")

    # Test Redis connection before starting
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # Test Redis connection
            redis_client.storage.ping()
            logger.info("Redis connection successful")
            break
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Failed to connect to Redis after all retries")
                return 1

    # Apply OpenCV optimizations
    optimize_opencv_performance()

    pathIn = "./app/video/people_walking_1080.mp4"

    # Check if video file exists
    if not os.path.exists(pathIn):
        logger.error(f"Video file does not exist: {pathIn}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        return 0

    # Use optimized video processor
    try:
        with VideoProcessor(pathIn, target_fps=1.0) as video_proc:
            performance_monitor = PerformanceMonitor("MotionDetection")

            # Real-time processing (simple and fast)
            last_image_resized = None
            processed_count = 0

            logger.info("Starting simple frame processing")

            for success, image in video_proc.read_frames():
                if not success:
                    break

                # Resize current image for motion detection (640px - good balance)
                resized_current = redis_client.resize_for_motion_detection(image)

                if last_image_resized is not None:
                    with timer("motion_detection"):
                        if motion_detection(last_image_resized, resized_current, threshold=30, min_motion_pixels=50):
                            # Send FULL RESOLUTION image immediately
                            with timer("stream_upload"):
                                redis_client.put_image_in_stream(image, "camera_001")

                # Store the resized version for next iteration (avoid re-resizing)
                last_image_resized = resized_current
                processed_count += 1
                performance_monitor.update()

                # Log progress periodically
                if processed_count % 10 == 0:
                    performance_monitor.log_stats()

            # Real-time processing complete (no remaining batch to process)

            # Final performance summary
            final_stats = performance_monitor.get_stats()
            logger.info(f"=== PROCESSING COMPLETE ===")
            logger.info(f"Total frames processed: {final_stats['frames_processed']}")
            logger.info(f"Total processing time: {final_stats['elapsed_time']:.2f} seconds")
            logger.info(f"Average processing speed: {final_stats['average_fps']:.2f} fps")

    except Exception as e:
        logger.error(f"Error in motion detection worker: {e}")
        return 1

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



