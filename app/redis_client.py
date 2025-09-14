import base64
import contextlib
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

    def __init__(self):
        # Basic connection setup - gets values from your k8s config
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))

        # Database 0 for data storage
        self.storage = redis.Redis(host=self.host, port=self.port, db=0, decode_responses=True)

        # Database 1 for streams (no decode_responses for binary data)
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
        """Put an image and camera string into the stream."""
        # Validate inputs
        if not isinstance(camera, str):
            raise TypeError("camera must be a string")

        # Handle different image input types
        if isinstance(image, np.ndarray):
            # OpenCV image array - encode as JPEG
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                raise ValueError("Failed to encode image as JPEG")
            image_bytes = buffer.tobytes()
        elif isinstance(image, (bytes, bytearray)):
            # Already bytes
            image_bytes = bytes(image)
        else:
            raise TypeError("image must be numpy array, bytes, or bytearray")

        # Encode image as base64 for Redis storage
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create stream item
        stream_data = {b"image_data": image_b64.encode("utf-8"), b"camera": camera.encode("utf-8")}

        # Add to stream and return the message ID
        message_id = self.streams.xadd(self.image_queue.encode("utf-8"), stream_data)
        return message_id.decode("utf-8")

    def take_image_from_stream(self, consumer_group=None, consumer_name=None, count=1, block=1000):
        """Take an image from the stream."""
        try:
            if consumer_group and consumer_name:
                # Use consumer group for reliable processing
                # First try to read pending messages
                try:
                    pending = self.streams.xreadgroup(
                        consumer_group.encode("utf-8"),
                        consumer_name.encode("utf-8"),
                        {self.image_queue.encode("utf-8"): b"0"},
                        count=count,
                        block=0,
                    )
                    if pending and pending[0][1]:
                        return self._parse_stream_messages(pending[0][1])
                except redis.exceptions.ResponseError:
                    # Consumer group doesn't exist, create it
                    with contextlib.suppress(redis.exceptions.ResponseError):
                        self.streams.xgroup_create(
                            self.image_queue.encode("utf-8"), consumer_group.encode("utf-8"), id=b"0", mkstream=True
                        )

                # Read new messages
                messages = self.streams.xreadgroup(
                    consumer_group.encode("utf-8"),
                    consumer_name.encode("utf-8"),
                    {self.image_queue.encode("utf-8"): b">"},
                    count=count,
                    block=block,
                )
            else:
                # Simple read without consumer group
                messages = self.streams.xread({self.image_queue.encode("utf-8"): b"$"}, count=count, block=block)

            if messages and messages[0][1]:
                return self._parse_stream_messages(messages[0][1])
            else:
                return []

        except redis.exceptions.ResponseError as e:
            if "NOGROUP" in str(e):
                # Consumer group doesn't exist
                return []
            raise

    @staticmethod
    def _parse_stream_messages(raw_messages):
        """Parse raw stream messages into usable format."""
        parsed_messages = []

        for message_id, fields in raw_messages:
            # Decode message ID
            msg_id = message_id.decode("utf-8")

            # Parse fields
            field_dict = {}
            for i in range(0, len(fields), 2):
                key = fields[i].decode("utf-8")
                value = fields[i + 1]

                if key == "image_data":
                    # Decode base64 image data back to bytes
                    image_b64 = value.decode("utf-8")
                    image_bytes = base64.b64decode(image_b64)
                    # Convert back to OpenCV image
                    image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    field_dict["image"] = image_array
                elif key == "camera":
                    field_dict["camera"] = value.decode("utf-8")

            parsed_messages.append({
                "message_id": msg_id,
                "image": field_dict.get("image"),
                "camera": field_dict.get("camera"),
            })

        return parsed_messages

    def acknowledge_message(self, consumer_group, message_id):
        """Acknowledge a message has been processed (when using consumer groups)."""
        if consumer_group:
            return self.streams.xack(
                self.image_queue.encode("utf-8"), consumer_group.encode("utf-8"), message_id.encode("utf-8")
            )
        return None


# Create one instance to use everywhere
redis_client = RedisClient()
