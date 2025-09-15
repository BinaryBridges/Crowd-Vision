"""
Performance optimization utilities for image processing and video analysis.
"""

import time
import logging
from contextlib import contextmanager
from typing import List, Tuple, Generator
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.debug(f"{operation_name} took {elapsed_time:.4f} seconds")


class VideoProcessor:
    """Optimized video processor with frame skipping and batching."""
    
    def __init__(self, video_path: str, target_fps: float = 1.0):
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = None
        self.original_fps = None
        self.frame_skip = 1
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        
        logger.info(f"Video FPS: {self.original_fps}, target FPS: {self.target_fps}, frame skip: {self.frame_skip}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
    
    def read_frames(self) -> Generator[Tuple[bool, np.ndarray], None, None]:
        """Generator that yields frames at the target FPS."""
        frame_count = 0
        
        while True:
            # Skip frames to achieve target FPS
            for _ in range(self.frame_skip):
                success, frame = self.cap.read()
                if not success:
                    return
                frame_count += 1
            
            yield True, frame


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.frame_count = 0
        self.last_log_time = self.start_time
        self.last_frame_count = 0
    
    def update(self, frames_processed: int = 1):
        """Update the frame count."""
        self.frame_count += frames_processed
    
    def log_stats(self, interval: float = 10.0):
        """Log performance statistics at specified intervals."""
        current_time = time.time()
        
        if current_time - self.last_log_time >= interval:
            elapsed_total = current_time - self.start_time
            elapsed_interval = current_time - self.last_log_time
            
            frames_in_interval = self.frame_count - self.last_frame_count
            
            avg_fps = self.frame_count / elapsed_total if elapsed_total > 0 else 0
            interval_fps = frames_in_interval / elapsed_interval if elapsed_interval > 0 else 0
            
            logger.info(f"{self.name} - Total: {self.frame_count} frames, "
                       f"Avg FPS: {avg_fps:.2f}, Recent FPS: {interval_fps:.2f}")
            
            self.last_log_time = current_time
            self.last_frame_count = self.frame_count
    
    def get_stats(self) -> dict:
        """Get current performance statistics."""
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            "frames_processed": self.frame_count,
            "elapsed_time": elapsed_time,
            "average_fps": avg_fps
        }


def optimize_opencv_performance():
    """Apply OpenCV performance optimizations."""
    # Limit OpenCV to single thread for better predictability in multi-process environments
    cv2.setNumThreads(1)
    
    # Set optimized buffer sizes
    cv2.setUseOptimized(True)
    
    logger.info("Applied OpenCV performance optimizations")