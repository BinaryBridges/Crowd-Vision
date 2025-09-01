from time import perf_counter

import cv2
import numpy as np

import config
from models import Status, TrackedFace

# Math helpers


def normalize_l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.array([])
    return a @ b


def reduce_embedding_for_tracking(full_embedding: np.ndarray) -> np.ndarray:
    reduced = full_embedding[: config.TRACK_VERIFY_VECTOR_SIZE]
    return normalize_l2(reduced)


# Drawing Helpers


def draw_text_label(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.6, thickness: int = 2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (x + 3, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
    )


def color_for_face_status(face: TrackedFace) -> tuple[int, int, int]:
    if face.status == Status.CONFIRMED_KEY:
        # Use category color for recognized persons; keep blue for confirmed Unknown
        if face.label != "Unknown":
            return _category_color(getattr(face, "category", None)) or (0, 200, 0)
        return (255, 0, 0)  # blue for confirmed unknown
    if face.status == Status.CONFIRMED_BAD:
        return _category_color(getattr(face, "category", None)) or (0, 0, 255)
    if face.status == Status.TENTATIVE:
        return (0, 255, 255)
    return (128, 128, 128)


def draw_tracked_faces(frame: np.ndarray, tracked: list[TrackedFace]):
    for face in tracked:
        if face.status == Status.LOST:
            continue

        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_for_face_status(face), 2)
        conf_txt = f" {face.confidence:.2f}" if getattr(face, "confidence", 0) > 0 else ""

        if face.status == Status.TENTATIVE:
            # still verifying
            if face.label == "Unknown":
                label = f"Unknown{conf_txt} (tentative)"
            else:
                label = f"maybe {face.label}{conf_txt} (tentative)"
        elif face.status in {Status.CONFIRMED_KEY, Status.CONFIRMED_BAD}:
            # confirmed state
            if face.label == "Unknown":
                label = "Unknown"
            else:
                role = getattr(face, "category", None)
                core = f"{face.label} [{role}]" if role else face.label
                label = f"{core}{conf_txt}"
        else:
            # fallback
            label = f"{face.label}{conf_txt}" if conf_txt else face.label

        draw_text_label(frame, label, x1, y1)


def expire_notifications(notified: dict[str, int], current_frame: int):
    """
    Remove notification entries whose cooldown has fully elapsed
    relative to the current frame.
    """
    for name, last_frame in list(notified.items()):
        # expire as soon as the full cooldown window has passed
        if current_frame - last_frame >= config.NOTIFY_COOLDOWN_FRAMES:
            notified.pop(name, None)


def _category_display_name(cat: str | None) -> str:
    if cat and hasattr(config, "CATEGORY_META") and cat in config.CATEGORY_META:
        return config.CATEGORY_META[cat].get("label") or cat
    return cat or "?"


def _category_color(cat: str | None) -> tuple[int, int, int] | None:
    if cat and hasattr(config, "CATEGORY_META") and cat in config.CATEGORY_META:
        color = config.CATEGORY_META[cat].get("color")
        if isinstance(color, (list, tuple)) and len(color) == 3:
            return tuple(int(x) for x in color)
    return None


class FPSMeter:
    """
    Lightweight FPS meter with EMA smoothing over frame time.
    Call tick() once per loop; read .fps for smoothed FPS and .inst_fps for instantaneous FPS.
    """

    def __init__(self, alpha: float = config.FPS_ALPHA, max_dt: float = config.FPS_MAX_DT_CLAMP):
        self.alpha = float(alpha)
        self.max_dt = float(max_dt)
        self._last_t = perf_counter()
        self._ema_dt = None
        self.fps = 0.0
        self.inst_fps = 0.0
        self.frames = 0

    def reset(self):
        self._last_t = perf_counter()
        self._ema_dt = None
        self.fps = 0.0
        self.inst_fps = 0.0
        self.frames = 0

    def tick(self):
        now = perf_counter()
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0:
            dt = 1e-6
        if self.max_dt is not None:
            dt = min(dt, self.max_dt)

        # instantaneous FPS
        self.inst_fps = 1.0 / dt

        # EMA over dt -> smoothed FPS = 1 / ema_dt
        if self._ema_dt is None:
            self._ema_dt = dt
        else:
            a = self.alpha
            self._ema_dt = (1.0 - a) * self._ema_dt + a * dt

        self.fps = 1.0 / self._ema_dt if self._ema_dt > 0 else self.inst_fps
        self.frames += 1


def draw_fps_label(frame, meter: FPSMeter, x: int = 10, y: int = 30):
    txt = f"{meter.fps:.{config.FPS_DECIMALS}f} FPS"
    draw_text_label(frame, txt, x, y)
