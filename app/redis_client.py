import os
from typing import ClassVar

import cv2
import numpy as np
import redis


class RedisClient:
    # === DEFINE YOUR DATA STRUCTURE ONCE ===
    FACE_FIELDS: ClassVar[list[str]] = ["embedding", "confidence", "age", "gender", "cameras"]

    # Stream item structure - image + camera string
    STREAM_ITEM_FIELDS: ClassVar[list[str]] = ["image_data", "camera"]

    @staticmethod
    def resize_for_motion_detection(image, max_width=None):
        """
        Resize image specifically for motion detection (optimized for speed).
        This should be used for motion detection only, not for images going to the stream.
        """
        if max_width is None:
            max_width = int(os.getenv("MOTION_DETECTION_RESIZE_WIDTH", "640"))

        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def __init__(self):
        # Basic connection setup - gets values from your k8s config
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))

        # Database 0 for data storage
        self.storage = redis.Redis(host=self.host, port=self.port, db=0, decode_responses=True)

        # Database 1 for streams (simple, fast connection)
        self.streams = redis.Redis(
            host=self.host, port=self.port, db=int(os.getenv("REDIS_DB_STREAMS", "1")), decode_responses=False
        )

        # Queue names from config
        self.image_queue = os.getenv("IMAGE_PROCESSOR_QUEUE", "image_queue")

    def save_face_data(self, **data):
        """Save face data using embedding as the key."""
        # Validate that only allowed fields are used
        for field in data:
            if field not in self.FACE_FIELDS:
                raise ValueError(f"Invalid field '{field}'. Allowed: {self.FACE_FIELDS}")

        # Use embedding as key, remove it from stored data
        key = f"face:{data['embedding']}"
        data_to_store = {k: v for k, v in data.items() if k != "embedding"}
        return self.storage.hset(key, mapping=data_to_store)

    def put_image_in_stream(self, image, camera):
        """
        Put an image into the stream - simple and fast.
        """
        # Get image properties
        height, width, channels = image.shape

        # Store raw numpy array data directly
        image_bytes = image.tobytes()

        # Store with metadata for reconstruction
        stream_data = {
            b"image_data": image_bytes,
            b"camera": camera.encode("utf-8"),
            b"height": str(height).encode("utf-8"),
            b"width": str(width).encode("utf-8"),
            b"channels": str(channels).encode("utf-8"),
            b"dtype": str(image.dtype).encode("utf-8"),
        }

        # Add to stream
        message_id = self.streams.xadd(self.image_queue.encode("utf-8"), stream_data)
        return message_id.decode("utf-8")

    def _ensure_consumer_group(self, groupname: str) -> None:
        """
        Create the consumer group if it does not exist.
        Uses id='0' so new workers can read any backlog already in the stream.
        """
        try:
            # mkstream=True will create the stream key if it doesn't exist yet.
            self.streams.xgroup_create(
                name=self.image_queue,
                groupname=groupname,
                id="0",
                mkstream=True,
            )
        except redis.exceptions.ResponseError as e:
            # Ignore if group already exists; re-raise anything else
            if "BUSYGROUP" not in str(e):
                raise

    def take_image_from_stream(self, consumer_group=None, consumer_name=None, count=1, block=1000):
        """
        Read images from Redis stream.
        Returns a list of dicts like:
        [{ "message_id": str, "image": np.ndarray, "camera": str }]
        """
        stream_key = self.image_queue  # str is fine; client encodes

        # If using groups, ensure group exists before reading
        if consumer_group and consumer_name:
            self._ensure_consumer_group(consumer_group)
            try:
                resp = self.streams.xreadgroup(
                    groupname=consumer_group,
                    consumername=consumer_name,
                    streams={stream_key: ">"},
                    count=count,
                    block=block,
                )
            except redis.exceptions.ResponseError as e:
                # In case of race where group was dropped between ensure & read
                if "NOGROUP" in str(e):
                    self._ensure_consumer_group(consumer_group)
                    resp = self.streams.xreadgroup(
                        groupname=consumer_group,
                        consumername=consumer_name,
                        streams={stream_key: ">"},
                        count=count,
                        block=block,
                    )
                else:
                    raise
        else:
            resp = self.streams.xread({stream_key: "0"}, count=count, block=block)

        if not resp:
            return []

        msgs = []
        for _key, entries in resp:
            for mid, fields in entries:
                try:
                    h = int(fields[b"height"].decode())
                    w = int(fields[b"width"].decode())
                    c = int(fields[b"channels"].decode())
                    dtype = np.dtype(fields[b"dtype"].decode())
                    img = np.frombuffer(fields[b"image_data"], dtype=dtype).reshape((h, w, c))
                    cam = fields[b"camera"].decode()
                    msgs.append({"message_id": mid.decode(), "image": img, "camera": cam})
                except (KeyError, ValueError, TypeError, UnicodeDecodeError) as e:
                    print(f"error reconstructing image: {e}", flush=True)
        return msgs

    def acknowledge_message(self, group: str, message_id: str) -> None:
        self.streams.xack(self.image_queue, group, message_id)

    def ping(self):
        """Test Redis connection."""
        return self.storage.ping()


# Create one instance to use everywhere
redis_client = RedisClient()
