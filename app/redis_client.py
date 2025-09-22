import base64
import os
import time
from typing import ClassVar

import cv2  # only used by resize_for_motion_detection
import numpy as np
import redis


class RedisClient:
    # === FACE HASH FIELDS ===
    FACE_FIELDS: ClassVar[list[str]] = [
        "embedding",
        "confidence",
        "age",
        "gender",
        "cameras",
        # new fields for similarity & stats
        "vec",  # base64 float32 reduced+normalized vector
        "last_seen",  # unix seconds
        "seen_count",  # int
    ]

    # Stream item structure - image + camera string
    STREAM_ITEM_FIELDS: ClassVar[list[str]] = ["image_data", "camera"]

    # ZSET index for recent faces (score = last_seen unix seconds)
    RECENT_ZSET: ClassVar[str] = "faces:seen"

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
        # Basic connection setup
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))

        # DB 0 for data storage
        self.storage = redis.Redis(host=self.host, port=self.port, db=0, decode_responses=True)

        # DB 1 for streams (binary payloads, so decode_responses=False)
        self.streams = redis.Redis(
            host=self.host, port=self.port, db=int(os.getenv("REDIS_DB_STREAMS", "1")), decode_responses=False
        )

        # Queue name
        self.image_queue = os.getenv("IMAGE_PROCESSOR_QUEUE", "image_queue")

    # ---------- legacy/simple face storage (still available) ----------
    def save_face_data(self, **data):
        """Save face data using embedding as the key (legacy helper)."""
        for field in data:
            if field not in self.FACE_FIELDS:
                raise ValueError(f"Invalid field '{field}'. Allowed: {self.FACE_FIELDS}")
        key = f"face:{data['embedding']}"
        data_to_store = {k: v for k, v in data.items() if k != "embedding"}
        return self.storage.hset(key, mapping=data_to_store)

    def face_exists(self, embedding_id: str) -> bool:
        return bool(self.storage.exists(f"face:{embedding_id}"))

    # ---------- image stream: producer ----------
    def put_image_in_stream(self, image, camera: str, *, frame: int | None = None, t_ns: int | None = None) -> str:
        """Serialize numpy image and push to Redis Stream."""
        height, width, channels = image.shape
        stream_data = {
            b"image_data": image.tobytes(),
            b"camera": camera.encode("utf-8"),
            b"height": str(height).encode("utf-8"),
            b"width": str(width).encode("utf-8"),
            b"channels": str(channels).encode("utf-8"),
            b"dtype": str(image.dtype).encode("utf-8"),
        }
        # optional metadata
        if frame is not None:
            stream_data[b"frame"] = str(int(frame)).encode("utf-8")
        if t_ns is not None:
            stream_data[b"t_ns"] = str(int(t_ns)).encode("utf-8")

        message_id = self.streams.xadd(self.image_queue.encode("utf-8"), stream_data)
        return message_id.decode("utf-8")

    # ---------- image stream: consumer ----------
    def _ensure_consumer_group(self, groupname: str) -> None:
        """Create consumer group if missing (id='0' to read backlog)."""
        try:
            self.streams.xgroup_create(
                name=self.image_queue,
                groupname=groupname,
                id="0",
                mkstream=True,
            )
        except redis.exceptions.ResponseError as e:
            # ignore "BUSYGROUP" (already exists)
            if "BUSYGROUP" not in str(e):
                raise

    def take_image_from_stream(self, consumer_group=None, consumer_name=None, count=1, block=1000):
        """
        Read images from Redis stream.
        Returns: list of dicts: [{ "message_id": str, "image": np.ndarray, "camera": str }]
        """
        stream_key = self.image_queue  # str is fine (client encodes internally)

        # XREADGROUP (reliable) or plain XREAD
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

        msgs: list[dict] = []
        for _key, entries in resp:
            for mid, fields in entries:
                try:
                    h = int(fields[b"height"].decode())
                    w = int(fields[b"width"].decode())
                    c = int(fields[b"channels"].decode())
                    dtype = np.dtype(fields[b"dtype"].decode())
                    img = np.frombuffer(fields[b"image_data"], dtype=dtype).reshape((h, w, c))
                    cam = fields[b"camera"].decode()

                    # optional metadata
                    frame = None
                    if b"frame" in fields:
                        try:
                            frame = int(fields[b"frame"].decode())
                        except (ValueError, TypeError, UnicodeDecodeError):
                            frame = None
                    t_ns = None
                    if b"t_ns" in fields:
                        try:
                            t_ns = int(fields[b"t_ns"].decode())
                        except (ValueError, TypeError, UnicodeDecodeError):
                            t_ns = None

                    msgs.append({
                        "message_id": mid.decode(),
                        "image": img,
                        "camera": cam,
                        "frame": frame,
                        "t_ns": t_ns,
                    })
                except (KeyError, ValueError, TypeError, UnicodeDecodeError) as e:
                    print(f'{{"event":"error","detail":"image_reconstruct","error":"{e}"}}', flush=True)
        return msgs

    def acknowledge_message(self, group: str, message_id: str) -> None:
        """Ack processed message (only when reading with consumer groups)."""
        self.streams.xack(self.image_queue, group, message_id)

    # ---------- similarity de-dupe utilities ----------
    @staticmethod
    def _vec_to_b64(vec: np.ndarray) -> str:
        return base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii")

    @staticmethod
    def _b64_to_vec(b64: str, dim: int) -> np.ndarray:
        raw = base64.b64decode(b64.encode("ascii"))
        return np.frombuffer(raw, dtype=np.float32).reshape(-1)[:dim]

    def record_new_face(
        self, *, embedding_id: str, vec: np.ndarray, age: str | None, gender: str | None, cameras: str
    ) -> None:
        """Store face payload + vector and index in recent ZSET."""
        now = int(time.time())
        face_key = f"face:{embedding_id}"
        mapping = {
            "confidence": "",
            "age": age or "",
            "gender": gender or "",
            "cameras": cameras,
            "vec": self._vec_to_b64(vec),
            "last_seen": str(now),
            "seen_count": "1",
        }
        self.storage.hset(face_key, mapping=mapping)
        self.storage.zadd(self.RECENT_ZSET, {embedding_id: now})

    def bump_existing_face(self, embedding_id: str) -> None:
        """Update last_seen/seen_count and recent index."""
        now = int(time.time())
        face_key = f"face:{embedding_id}"
        pipe = self.storage.pipeline()
        pipe.hincrby(face_key, "seen_count", 1)
        pipe.hset(face_key, mapping={"last_seen": str(now)})
        pipe.zadd(self.RECENT_ZSET, {embedding_id: now})
        pipe.execute()

    def find_similar_face(self, vec: np.ndarray, *, threshold: float = 0.92, recent: int = 200) -> str | None:
        """
        Compare against the most recent N faces using cosine similarity.
        Returns embedding_id if a match ≥ threshold is found, else None.
        """
        ids = self.storage.zrevrange(self.RECENT_ZSET, 0, recent - 1)
        if not ids:
            return None

        dim = vec.shape[-1]
        # ensure L2 normalized (just in case caller didn't)
        vec = vec / (np.linalg.norm(vec) + 1e-9)  # noqa: PLR6104

        # batch HMGET to reduce round-trips
        for i in range(0, len(ids), 50):
            chunk = ids[i : i + 50]
            keys = [f"face:{fid}" for fid in chunk]
            pipe = self.storage.pipeline()
            for k in keys:
                pipe.hget(k, "vec")
            vec_b64_list = pipe.execute()

            for fid, v64 in zip(chunk, vec_b64_list, strict=False):
                if not v64:
                    continue
                cand = self._b64_to_vec(v64, dim)
                cand = cand / (np.linalg.norm(cand) + 1e-9)  # noqa: PLR6104
                sim = float(np.dot(vec, cand))
                if sim >= threshold:
                    return fid  # found duplicate
        return None

    # Guard for exact-ID races across workers
    def save_face_if_new(self, *, embedding_id: str, age: str | None, gender: str | None, cameras: str) -> bool:
        """
        Atomic id-based guard (race-safety across workers).
        Returns True if this process created the entry, False if existed.
        """
        guard_key = f"face:{embedding_id}:created"
        face_key = f"face:{embedding_id}"
        if self.storage.setnx(guard_key, "1"):
            mapping = {"confidence": "", "age": age or "", "gender": gender or "", "cameras": cameras}
            self.storage.hset(face_key, mapping=mapping)
            return True
        return False

    def ping(self):
        """Test Redis connection."""
        return self.storage.ping()


# Singleton
redis_client = RedisClient()
