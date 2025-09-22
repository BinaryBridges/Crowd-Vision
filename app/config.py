import os
import socket

DETECTION_MODEL_NAME = "buffalo_s"
DET_SIZE = (640, 640)
TRACK_VERIFY_VECTOR_SIZE = 128
IMAGE_QUEUE = os.getenv("IMAGE_PROCESSOR_QUEUE", "image_queue")
REDIS_CONSUMER_GROUP = os.getenv("REDIS_CONSUMER_GROUP", "imgproc")
REDIS_CONSUMER_NAME = os.getenv("REDIS_CONSUMER_NAME", socket.gethostname()[:20] or "worker")

LOG_JSON = os.getenv("LOG_JSON", "1") == "1"

DEDUPE_SIM_THRESHOLD = float(os.getenv("DEDUPE_SIM_THRESHOLD", "0.60"))
RECENT_FACE_WINDOW = int(os.getenv("RECENT_FACE_WINDOW", "200"))
