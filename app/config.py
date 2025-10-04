import os
import socket

DETECTION_MODEL_NAME = "buffalo_s"
DET_SIZE = (640, 640)
VECTOR_SIZE = 256
IMAGE_QUEUE = os.getenv("IMAGE_PROCESSOR_QUEUE", "image_queue")
REDIS_CONSUMER_GROUP = os.getenv("REDIS_CONSUMER_GROUP", "imgproc")
REDIS_CONSUMER_NAME = os.getenv("REDIS_CONSUMER_NAME", socket.gethostname()[:20] or "worker")

LOG_JSON = os.getenv("LOG_JSON", "1") == "1"

DEDUPE_SIM_THRESHOLD = float(os.getenv("DEDUPE_SIM_THRESHOLD", "0.60"))

# Moving average configuration
# Update face data (vector and age) every N times we see the same face
FACE_UPDATE_INTERVAL = int(os.getenv("FACE_UPDATE_INTERVAL", "10"))

# RedisSearch configuration (only search method used)
REDISEARCH_INDEX_NAME = os.getenv("REDISEARCH_INDEX_NAME", "faces_idx")

# Redis Database Separation for Optimal Performance:
# DB 0: Face embeddings, demographics, vectors (persistent data for search)
# DB 1: Image processing streams (temporary data)
# DB 2: Video processing metadata (locks, completion status)
# DB 3: Face deduplication guards and similarity metadata (cleanup-safe)

# Convex Database Integration
CONVEX_BASE_URL = os.getenv("CONVEX_BASE_URL", "http://127.0.0.1:3210")
CONVEX_USER_ID = os.getenv("CONVEX_USER_ID", "j97bnd4prz3ecpw2fcat9ks5yn7rmah0")
CONVEX_EVENT_NAME = os.getenv("CONVEX_EVENT_NAME", "Face Detection Event")
CONVEX_EVENT_PRICE = float(os.getenv("CONVEX_EVENT_PRICE", "0.0"))
