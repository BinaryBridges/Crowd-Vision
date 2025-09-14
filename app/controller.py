import json
import os

from app import config
from app.redis_client import redis_client
from app.utils import load_bgr_image


def start_controller():
    path = os.getenv("SAMPLE_IMAGE", config.SAMPLE_IMAGE)
    img = load_bgr_image(path)
    msg_id = redis_client.put_image_in_stream(img, camera="sample")
    print(json.dumps({"event": "controller_enqueued_sample", "message_id": msg_id}), flush=True)


def reconcile():
    pass


def shutdown():
    pass
