import os
import socket
import time
from typing import ClassVar

import cv2  # only used by resize_for_motion_detection
import numpy as np
import redis
from redis.exceptions import ResponseError

from app import config

# Try to import redis-py's Search helpers; fall back to raw commands if missing
try:
    from redis.commands.search.field import NumericField, TextField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    SEARCH_API_AVAILABLE = True
except (ImportError, AttributeError):
    SEARCH_API_AVAILABLE = False
    VectorField = TextField = NumericField = None
    IndexDefinition = IndexType = Query = None


class RedisClient:
    # === FACE HASH FIELDS ===
    FACE_FIELDS: ClassVar[list[str]] = [
        "embedding",
        "confidence",
        "age",
        "gender",
        "cameras",
        # new fields for similarity & stats
        "vec",  # base64/bytes float32 reduced+normalized vector
        "last_seen",  # unix seconds
        "seen_count",  # int
    ]

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
        # Basic connection setup
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))

        # Connection configuration
        self.connection_config = {
            "host": self.host,
            "port": self.port,
            "socket_connect_timeout": 5,
            "socket_timeout": 5,
            "retry_on_timeout": True,
            "health_check_interval": 30,
        }

        # Initialize connections with retry logic
        self.storage = None
        self.streams = None
        self._initialize_connections()

        # Queue name
        self.image_queue = os.getenv("IMAGE_PROCESSOR_QUEUE", "image_queue")

        # Search / index config
        self.index_name = config.REDISEARCH_INDEX_NAME
        self.vector_dim = config.VECTOR_SIZE
        self.use_redisearch = True  # keep feature enabled

        # Verify that Search commands exist on the server (Redis 8 has them built-in)
        if not self._has_search_commands():
            raise RuntimeError(
                "Redis Search commands (FT.*) are not available on this server. "
                "Use Redis 8+ (built-in Query) or Redis 7.x with RediSearch enabled."
            )

        # Initialize index (with retry)
        self._ensure_search_index_with_retry()

    # ---- Utilities ----
    def _has_search_commands(self) -> bool:
        """
        Detect Search availability by checking for FT.SEARCH using COMMAND INFO.
        Returns True if the command exists, False otherwise.
        """

        def _result_has_payload(result):
            if result is None:
                return False
            if isinstance(result, dict):
                return any(_result_has_payload(v) for v in result.values())
            if isinstance(result, (list, tuple, set)):
                return any(_result_has_payload(v) for v in result)
            return bool(result)

        try:
            # Works across RESP2/RESP3 by looking for any payload in the reply.
            info = self.storage.execute_command("COMMAND", "INFO", "FT.SEARCH")
            return _result_has_payload(info)
        except (ResponseError, redis.exceptions.RedisError):
            return False

    def _initialize_connections(self) -> None:
        """Initialize Redis connections with robust retry logic."""
        print("Initializing Redis connections...")

        # Initialize storage connection (DB 0)
        self.storage = self._create_connection_with_retry(db=0, decode_responses=True, description="storage")

        # Initialize streams connection (DB 1)
        streams_db = int(os.getenv("REDIS_DB_STREAMS", "1"))
        self.streams = self._create_connection_with_retry(db=streams_db, decode_responses=False, description="streams")

        print("Redis connections initialized successfully")

    def _create_connection_with_retry(
        self, db: int, *, decode_responses: bool, description: str
    ) -> redis.Redis:
        """Create a Redis connection with exponential backoff retry logic."""
        max_retries = int(os.getenv("REDIS_MAX_RETRIES", "30"))  # Up to 30 retries
        base_delay = float(os.getenv("REDIS_BASE_DELAY", "1.0"))  # Start with 1 second
        max_delay = float(os.getenv("REDIS_MAX_DELAY", "30.0"))  # Cap at 30 seconds

        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to Redis {description} (attempt {attempt + 1}/{max_retries})...")
                print(f"  Target: {self.host}:{self.port} (DB {db})")

                # Test basic network connectivity first
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((self.host, self.port))
                    sock.close()

                    if result != 0:
                        msg = f"Cannot reach {self.host}:{self.port} (network error: {result})"
                        raise ConnectionError(msg)

                    print(f"  Network connectivity to {self.host}:{self.port}: OK")

                except OSError as e:
                    raise ConnectionError(f"Network connectivity test failed: {e}") from e

                # Create Redis connection with conservative settings
                connection = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=db,
                    socket_connect_timeout=10,  # Increased timeout
                    socket_timeout=10,  # Increased timeout
                    socket_keepalive=True,  # Enable keepalive
                    socket_keepalive_options={},
                    retry_on_timeout=True,
                    decode_responses=decode_responses,
                    max_connections=10,  # Reduced pool size
                )

                # Test the connection
                print("  Testing Redis ping...")
                pong = connection.ping()
                print(f"  Redis ping response: {pong}")

                print(f"Successfully connected to Redis {description}")
                return connection

            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, OSError, BrokenPipeError) as e:
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff, but use shorter delays for startup issues
                    if "Broken pipe" in str(e) or "Connection refused" in str(e):
                        # Redis is starting up, use shorter delays
                        delay = min(base_delay * (1.5**attempt), 10.0)  # Cap at 10s for startup issues
                        print(f"Redis {description} is starting up (attempt {attempt + 1}): {e}")
                    else:
                        # Other connection issues, use normal backoff
                        delay = min(base_delay * (2**attempt), max_delay)
                        print(f"Redis {description} connection failed (attempt {attempt + 1}): {e}")

                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue

                print(f"Failed to connect to Redis {description} after {max_retries} attempts")
                raise redis.exceptions.ConnectionError(
                    f"Could not connect to Redis {description} at {self.host}:{self.port} after {max_retries} attempts"
                ) from e
            except Exception as e:
                # Handle other unexpected errors
                error_msg = str(e).lower()
                if any(
                    keyword in error_msg
                    for keyword in ["broken pipe", "connection reset", "connection refused", "timeout"]
                ):
                    # These are likely startup-related, treat like connection errors
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (1.5**attempt), 10.0)
                        print(f"Redis {description} startup issue (attempt {attempt + 1}): {e}")
                        print(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue

                    print(f"Failed to connect to Redis {description} after {max_retries} attempts")
                    raise redis.exceptions.ConnectionError(
                        f"Could not connect to Redis {description} at {self.host}:{self.port} after {max_retries} attempts"
                    ) from e

                # Truly unexpected error, re-raise immediately
                print(f"Unexpected error connecting to Redis {description}: {e}")
                raise

        # If we get here, all retries failed
        msg = f"Could not connect to Redis {description} at {self.host}:{self.port} after {max_retries} attempts"
        raise redis.exceptions.ConnectionError(msg)

    def _ensure_search_index_with_retry(self) -> None:
        """Ensure Redis Search index exists with retry logic."""
        max_retries = 10
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                self._ensure_search_index()
                return  # Success
            except (redis.exceptions.RedisError, RuntimeError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(f"Search index creation failed (attempt {attempt + 1}): {e}")
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue

                print(f"Failed to create Search index after {max_retries} attempts.")
                raise RuntimeError("Search index creation failed after all retry attempts") from e

    # ---------- legacy/simple face storage (still available) ----------
    def save_face_data(self, **data):
        """Save face data using embedding as the key (legacy helper)."""
        for field in data:
            if field not in self.FACE_FIELDS:
                raise ValueError(f"Invalid field '{field}'. Allowed: {self.FACE_FIELDS}")
        key = data["embedding"]  # Clean key since DB 0 is exclusive to face data
        data_to_store = {k: v for k, v in data.items() if k != "embedding"}
        return self.storage.hset(key, mapping=data_to_store)

    def face_exists(self, embedding_id: str) -> bool:
        return bool(self.storage.exists(embedding_id))  # Clean key

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

    def acknowledge_and_delete_message(self, group: str, message_id: str) -> None:
        """
        Acknowledge and delete a processed message from the stream.

        This is the proper way to handle message completion:
        1. XACK - Acknowledge the message (removes from pending)
        2. XDEL - Delete the message from the stream (frees memory)

        Args:
            group: Consumer group name
            message_id: Message ID to acknowledge and delete

        """
        try:
            # First acknowledge the message (removes from pending list)
            ack_result = self.streams.xack(self.image_queue, group, message_id)

            # Then delete the message from the stream (frees memory)
            del_result = self.streams.xdel(self.image_queue, message_id)

            if ack_result and del_result:
                # Successfully acknowledged and deleted
                pass
            elif not ack_result:
                print(f'{{"event":"warning","detail":"message_already_acked","message_id":"{message_id}"}}', flush=True)
            elif not del_result:
                print(
                    f'{{"event":"warning","detail":"message_already_deleted","message_id":"{message_id}"}}', flush=True
                )

        except redis.exceptions.RedisError as e:
            print(
                f'{{"event":"error","detail":"message_cleanup_failed","message_id":"{message_id}","error":"{e}"}}',
                flush=True,
            )
            raise

    def cleanup_old_messages(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old processed messages from the stream.

        This removes messages older than max_age_seconds that have been
        acknowledged by all consumer groups.

        Args:
            max_age_seconds: Maximum age of messages to keep (default: 1 hour)

        Returns:
            Number of messages removed

        """
        try:
            # Calculate timestamp threshold (Redis uses milliseconds)
            current_time_ms = int(time.time() * 1000)
            threshold_ms = current_time_ms - (max_age_seconds * 1000)

            # Use XTRIM with MINID to remove old messages
            # This keeps messages newer than the threshold
            removed_count = self.streams.xtrim(
                self.image_queue,
                minid=f"{threshold_ms}-0",
                approximate=True,  # More efficient, allows some variance
            )

            if removed_count > 0:
                print(
                    f'{{"event":"info","detail":"stream_cleanup","removed_messages":{removed_count},"max_age_seconds":{max_age_seconds}}}',
                    flush=True,
                )
            else:
                return removed_count

        except redis.exceptions.RedisError as e:
            print(f'{{"event":"error","detail":"stream_cleanup_failed","error":"{e}"}}', flush=True)
            return 0

    # ---------- Search / vector utilities ----------
    @staticmethod
    def _bytes_to_vec(vec_bytes: bytes, dim: int) -> np.ndarray:
        """Convert bytes directly to vector (Redis vector binary format)."""
        return np.frombuffer(vec_bytes, dtype=np.float32).reshape(-1)[:dim]

    def record_new_face(
        self, *, embedding_id: str, vec: np.ndarray, age: str | None, gender: str | None, cameras: str
    ) -> None:
        """Store face payload + vector in Redis hash; vector as FLOAT32 bytes."""
        now = int(time.time())
        face_key = embedding_id  # Clean key since DB 0 is exclusive to face data

        mapping = {
            "confidence": "",
            "age": age or "",
            "gender": gender or "",
            "cameras": cameras,
            "last_seen": str(now),
            "seen_count": "1",
            "vec": vec.astype(np.float32).tobytes(),  # binary blob
        }

        self.storage.hset(face_key, mapping=mapping)

    def bump_existing_face(self, embedding_id: str) -> int:
        """
        Update last_seen/seen_count.
        Returns the new seen_count value.
        """
        now = int(time.time())
        face_key = embedding_id  # Clean key since DB 0 is exclusive to face data
        pipe = self.storage.pipeline()
        pipe.hincrby(face_key, "seen_count", 1)
        pipe.hset(face_key, mapping={"last_seen": str(now)})
        results = pipe.execute()
        # hincrby returns the new value
        return int(results[0])

    def get_face_data(self, embedding_id: str) -> dict | None:
        """
        Retrieve face data including vector and age.
        Returns dict with 'vec' (numpy array), 'age' (str), 'seen_count' (int), or None if not found.
        """
        face_key = embedding_id  # Clean key since DB 0 is exclusive to face data
        data = self.storage.hgetall(face_key)
        if not data:
            return None

        result = {
            "age": data.get("age", ""),
            "seen_count": int(data.get("seen_count", "1")),
        }

        # Decode the vector if it exists
        vec_bytes = data.get("vec")
        if vec_bytes:
            # vec is stored as FLOAT32 bytes
            vec_array = np.frombuffer(
                vec_bytes.encode("latin1") if isinstance(vec_bytes, str) else vec_bytes, dtype=np.float32
            )
            result["vec"] = vec_array

        return result

    def update_face_with_moving_average(
        self, *, embedding_id: str, new_vec: np.ndarray, new_age: int | None, seen_count: int
    ) -> None:
        """
        Update face data with moving average of vector and age.

        Moving average formula: new_avg = (old_avg * (n-1) + new_value) / n
        where n is the seen_count
        """
        face_key = embedding_id  # Clean key since DB 0 is exclusive to face data

        # Get current data
        current_data = self.get_face_data(embedding_id)
        if not current_data:
            return

        # Calculate moving average for vector
        current_vec = current_data.get("vec")
        if current_vec is not None and len(current_vec) == len(new_vec):
            # Moving average: avg_new = (avg_old * (n-1) + new_value) / n
            avg_vec = (current_vec * (seen_count - 1) + new_vec) / seen_count
            # Re-normalize to unit length
            avg_vec /= np.linalg.norm(avg_vec) + 1e-9
        else:
            avg_vec = new_vec

        # Calculate moving average for age
        current_age_str = current_data.get("age", "")
        if current_age_str and new_age is not None:
            try:
                current_age = int(current_age_str)
                # Moving average for age
                avg_age = round((current_age * (seen_count - 1) + new_age) / seen_count)
            except (ValueError, TypeError):
                avg_age = new_age
        elif new_age is not None:
            avg_age = new_age
        else:
            avg_age = None

        # Update the face data
        mapping = {
            "vec": avg_vec.astype(np.float32).tobytes(),
        }
        if avg_age is not None:
            mapping["age"] = str(avg_age)

        self.storage.hset(face_key, mapping=mapping)

    def _ensure_search_index(self) -> None:
        """Create vector index for faces if it doesn't exist."""
        try:
            # If redis-py's Search API is available, try info() first
            if SEARCH_API_AVAILABLE:
                self.storage.ft(self.index_name).info()
                return
            else:
                # Raw check: FT.INFO <index>
                self.storage.execute_command("FT.INFO", self.index_name)
                return
        except redis.exceptions.ResponseError:
            # Index doesn't exist; create it below
            pass

        # Define schema / issue FT.CREATE
        if SEARCH_API_AVAILABLE:
            # Use redis-py search helpers
            schema = [
                VectorField(
                    "vec",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                TextField("age"),
                TextField("gender"),
                TextField("cameras"),
                NumericField("last_seen"),
                NumericField("seen_count"),
            ]
            # No prefix needed since DB 0 is exclusive to face data
            definition = IndexDefinition(index_type=IndexType.HASH)
            self.storage.ft(self.index_name).create_index(schema, definition=definition)
        else:
            # Raw FT.CREATE without prefix since DB 0 is exclusive to face data
            self.storage.execute_command(
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "SCHEMA",
                "vec",
                "VECTOR",
                "FLAT",
                6,
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.vector_dim),
                "DISTANCE_METRIC",
                "COSINE",
                "age",
                "TEXT",
                "gender",
                "TEXT",
                "cameras",
                "TEXT",
                "last_seen",
                "NUMERIC",
                "seen_count",
                "NUMERIC",
            )
        print(f"Created Search index: {self.index_name}")

    @staticmethod
    def _parse_raw_search_response(resp, threshold: float):
        """
        Parse raw FT.SEARCH response and find matching document.
        Returns docid if a match ≥ threshold is found, else None.
        """
        # resp[0] = total, then alternating [docid, [field, value, ...]]
        if not resp or len(resp) < 2:
            return None

        i = 1
        while i < len(resp):
            docid = resp[i].decode() if isinstance(resp[i], (bytes, bytearray)) else str(resp[i])
            fields = resp[i + 1]
            # fields like [b'score', b'0.1234']
            score = RedisClient._extract_score_from_fields(fields)
            if score is not None:
                similarity = 1.0 - score
                if similarity >= threshold:
                    return docid  # Clean key, no prefix to remove
            i += 2
        return None

    @staticmethod
    def _extract_score_from_fields(fields):
        """Extract score value from Redis search response fields."""
        if not isinstance(fields, list):
            return None

        for j in range(0, len(fields), 2):
            if fields[j] in {b"score", "score"}:
                val = fields[j + 1]
                return float(val.decode() if isinstance(val, (bytes, bytearray)) else val)
        return None

    def _search_with_api(self, vec_bytes: bytes, limit: int, threshold: float) -> str | None:
        """Search using redis-py Search API (DIALECT 2)."""
        try:
            query = (
                Query(f"(*)=>[KNN {limit} @vec $vec AS score]")
                .sort_by("score")
                .return_fields("score")
                .paging(0, limit)
                .dialect(2)
            )
            results = self.storage.ft(self.index_name).search(
                query,
                query_params={"vec": vec_bytes},
            )
            for doc in results.docs:
                distance = float(doc.score)
                similarity = 1.0 - distance
                if similarity >= threshold:
                    return doc.id  # Clean key, no prefix to remove
            return None
        except redis.exceptions.ResponseError as e:
            print(f"Search query error: {e}")
            raise

    def _search_with_raw_command(self, vec_bytes: bytes, limit: int, threshold: float) -> str | None:
        """Search using raw FT.SEARCH command (DIALECT 2)."""
        try:
            resp = self.storage.execute_command(
                "FT.SEARCH",
                self.index_name,
                f"(*)=>[KNN {limit} @vec $qv AS score]",
                "PARAMS",
                2,
                "qv",
                vec_bytes,
                "SORTBY",
                "score",
                "RETURN",
                1,
                "score",
                "LIMIT",
                0,
                limit,
                "DIALECT",
                2,
            )
            return self._parse_raw_search_response(resp, threshold)
        except redis.exceptions.ResponseError as e:
            print(f"Search query error: {e}")
            raise
        except (ValueError, TypeError) as e:
            print(f"Unexpected Search error: {e}")
            raise

    def find_similar_face_redisearch(self, vec: np.ndarray, *, threshold: float = 0.92, limit: int = 10) -> str | None:
        """
        Use vector similarity (cosine) to find matching faces.
        Returns embedding_id if a match ≥ threshold is found, else None.
        """
        # Ensure L2 normalized
        vec /= np.linalg.norm(vec) + 1e-9
        vec_bytes = vec.astype(np.float32).tobytes()

        if SEARCH_API_AVAILABLE:
            return self._search_with_api(vec_bytes, limit, threshold)
        else:
            return self._search_with_raw_command(vec_bytes, limit, threshold)

    # DEPRECATED: Guard for exact-ID races across workers (legacy method)
    def save_face_if_new(self, *, embedding_id: str, age: str | None, gender: str | None, cameras: str) -> bool:
        """
        DEPRECATED: Use app.utils.save_face_if_new() instead for separated storage.

        This legacy method stores both face data and guards in DB 0, which can
        impact search performance. The new separated storage approach stores:
        - Face data in DB 0 (for optimal search)
        - Deduplication guards in DB 3 (for safe cleanup)
        """
        guard_key = f"{embedding_id}:created"  # Legacy format with suffix
        face_key = embedding_id  # Clean key
        if self.storage.setnx(guard_key, "1"):
            mapping = {"confidence": "", "age": age or "", "gender": gender or "", "cameras": cameras}
            self.storage.hset(face_key, mapping=mapping)
            return True
        return False

    def ping(self):
        """Test Redis connection."""
        return self.storage.ping()

    def wait_for_redis(self, timeout: int = 300) -> bool:
        """Wait for Redis to be ready with comprehensive health checks."""
        def _fail(msg: str) -> None:
            # Local helper to satisfy the linters “abstract raise” rule
            raise redis.exceptions.RedisError(msg)

        print(f"Waiting for Redis to be ready (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Test basic connectivity
                self.storage.ping()
                self.streams.ping()

                # Test basic operations
                test_key = "health_check_test"
                self.storage.set(test_key, "test_value", ex=10)
                value = self.storage.get(test_key)
                self.storage.delete(test_key)

                # Handle bytes vs str (decode_responses may be False)
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")

                if value != "test_value":
                    _fail("Basic operations test failed")

                # If Search is enabled, ensure index exists/ready
                if self.use_redisearch:
                    try:
                        self._ensure_search_index()
                    except (redis.exceptions.RedisError, RuntimeError) as e:
                        print(f"Search not ready yet: {e}")
                        time.sleep(2)
                        continue

                print("Redis is ready and healthy!")
                return True

            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                print(f"Redis not ready yet ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining): {e}")
                time.sleep(min(5, max(0, remaining)))

            except redis.exceptions.RedisError as e:
                print(f"Unexpected error during Redis health check: {e}")
                time.sleep(2)

        print(f"Redis failed to become ready within {timeout} seconds")
        return False

    # ---------- Worker Status & Completion Detection ----------
    def register_worker(self, worker_id: str, worker_type: str) -> None:
        """Register a worker and set initial heartbeat."""
        now = int(time.time())
        worker_key = f"worker:heartbeat:{worker_id}"
        worker_data = {
            "type": worker_type,
            "status": "active",
            "last_seen": str(now),
            "started_at": str(now),
        }
        # Set with 90 second TTL (workers should update every 30s)
        self.storage.hset(worker_key, mapping=worker_data)
        self.storage.expire(worker_key, 90)
        print(f"Registered worker: {worker_id} ({worker_type})")

    def update_worker_heartbeat(self, worker_id: str, status: str = "active") -> None:
        """Update worker heartbeat and status."""
        now = int(time.time())
        worker_key = f"worker:heartbeat:{worker_id}"
        try:
            # Only update if key exists (worker was registered)
            if self.storage.exists(worker_key):
                self.storage.hset(
                    worker_key,
                    mapping={
                        "status": status,
                        "last_seen": str(now),
                    },
                )
                self.storage.expire(worker_key, 90)
        except redis.exceptions.RedisError as e:
            print(f"Failed to update heartbeat for worker {worker_id}: {e}")

    def unregister_worker(self, worker_id: str) -> None:
        """Remove worker heartbeat when worker exits gracefully."""
        worker_key = f"worker:heartbeat:{worker_id}"
        try:
            self.storage.delete(worker_key)
            print(f"Unregistered worker: {worker_id}")
        except redis.exceptions.RedisError as e:
            print(f"Failed to unregister worker {worker_id}: {e}")

    def get_active_workers(self, *, stale_threshold: int = 60) -> dict:
        """
        Get all currently active workers.

        Args:
            stale_threshold: Consider workers stale if not seen in this many seconds (default: 60)

        Returns:
            Dictionary of active (non-stale) workers

        """
        workers = {}
        now = int(time.time())
        try:
            # Find all worker heartbeat keys
            worker_keys = self.storage.keys("worker:heartbeat:*")
            for key in worker_keys:
                worker_data = self.storage.hgetall(key)
                if worker_data:
                    last_seen = int(worker_data.get("last_seen", "0"))
                    # Skip stale workers (haven't updated heartbeat recently)
                    if now - last_seen > stale_threshold:
                        continue

                    worker_id = key.replace("worker:heartbeat:", "")
                    workers[worker_id] = {
                        "type": worker_data.get("type", "unknown"),
                        "status": worker_data.get("status", "unknown"),
                        "last_seen": last_seen,
                        "started_at": int(worker_data.get("started_at", "0")),
                    }
        except redis.exceptions.RedisError as e:
            print(f"Failed to get active workers: {e}")
        return workers

    def get_stream_status(self) -> dict:
        """Get current status of the image processing stream."""
        try:
            # Get stream length (total messages in stream)
            stream_length = self.streams.xlen(self.image_queue)

            # Get pending messages info for the consumer group
            pending_info = {"pending": 0, "consumers": 0}
            try:
                # Check if consumer group exists and get pending info
                group_info = self.streams.xpending(self.image_queue, "imgproc")
                if group_info:
                    pending_info["pending"] = group_info.get("pending", 0)
                    # Get consumer count
                    consumers = self.streams.xinfo_consumers(self.image_queue, "imgproc")
                    pending_info["consumers"] = len(consumers) if consumers else 0
            except redis.exceptions.ResponseError:
                # Consumer group doesn't exist yet
                pass

            # Stream is idle when:
            # 1. No new messages in the stream (stream_length == 0)
            # 2. No messages pending processing (pending == 0)
            # With proper message deletion, stream_length should stay low
            is_idle = stream_length == 0 and pending_info["pending"] == 0

            return {
                "stream_length": stream_length,
                "pending_messages": pending_info["pending"],
                "active_consumers": pending_info["consumers"],
                "is_idle": is_idle,
                "total_unprocessed": stream_length + pending_info["pending"],  # Total work remaining
            }
        except redis.exceptions.RedisError as e:
            print(f"Failed to get stream status: {e}")
            return {
                "stream_length": -1,
                "pending_messages": -1,
                "active_consumers": -1,
                "is_idle": False,
                "total_unprocessed": -1,
            }

    def check_system_completion(self) -> dict:
        """
        Check if the entire system has completed all work.

        Returns:
            dict with completion status and details

        """
        from app.motion_detection_worker import list_video_status  # noqa: PLC0415

        # Check video processing status
        video_status = list_video_status()
        videos_complete = all(status == "completed" for status in video_status.values()) if video_status else True
        available_videos = sum(1 for status in video_status.values() if status == "available")
        locked_videos = sum(1 for status in video_status.values() if status == "locked")
        completed_videos = sum(1 for status in video_status.values() if status == "completed")

        # Check stream processing status
        stream_status = self.get_stream_status()
        stream_idle = stream_status["is_idle"]

        # Check worker status
        workers = self.get_active_workers()
        motion_workers = {k: v for k, v in workers.items() if v["type"] == "motion-detection"}
        image_workers = {k: v for k, v in workers.items() if v["type"] == "image-processor"}

        # System is complete when:
        # 1. All videos are processed (no available or locked videos)
        # 2. Image stream is empty and no pending messages
        # 3. No motion detection workers are active (they exit when done)
        system_complete = videos_complete and stream_idle and len(motion_workers) == 0

        return {
            "system_complete": system_complete,
            "videos": {
                "total": len(video_status),
                "completed": completed_videos,
                "available": available_videos,
                "locked": locked_videos,
                "all_complete": videos_complete,
            },
            "stream": stream_status,
            "workers": {
                "motion_detection": len(motion_workers),
                "image_processor": len(image_workers),
                "total_active": len(workers),
            },
            "details": {"video_status": video_status, "active_workers": workers},
        }


# Singleton
redis_client = RedisClient()
