import contextlib
from time import perf_counter

import cv2
import numpy as np

import config


# Math helpers
def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _l2(np.asarray(a, dtype=np.float32))
    b = _l2(np.asarray(b, dtype=np.float32))
    return a @ b.T


def reduce_embedding_for_tracking(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    k = min(config.TRACK_VERIFY_VECTOR_SIZE, emb.shape[-1])
    return _l2(emb[..., :k])


def _short_gender(g) -> str | None:
    # Accepts numeric or string (varies by model)
    if isinstance(g, (int, np.integer)):
        return "M" if int(g) == 1 else "F"
    if isinstance(g, str):
        s = g.lower()
        if s.startswith("m"):
            return "M"
        if s.startswith("f"):
            return "F"
    return None


def _fmt_demo_from_face(face_obj) -> str:
    # Try both 'gender' and 'sex' attributes; try 'age'
    g = getattr(face_obj, "gender", None)
    if g is None:
        g = getattr(face_obj, "sex", None)
    gender_short = _short_gender(g)

    age_val = getattr(face_obj, "age", None)
    parts = []
    if gender_short:
        parts.append(gender_short)
    if age_val is not None:
        with contextlib.suppress(Exception):
            parts.append(str(round(float(age_val))))

    return (" (" + ", ".join(parts) + ")") if parts else ""


def _label(img, text: str, x: int, y: int) -> None:
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), config.TEXT_BG, -1)
    cv2.putText(img, text, (x, y), font, scale, config.TEXT_COLOR, thickness, cv2.LINE_AA)


def draw_detections(frame, detections) -> None:
    """
    Draw boxes + demographics for the current frame's detections only (stateless).
    """
    for f in detections:
        # bbox comes from insightface face object
        x1, y1, x2, y2 = map(int, f.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_UNKNOWN, 2)
        _label(frame, "Unknown" + _fmt_demo_from_face(f), x1 + 2, max(20, y1 - 6))


# FPS
class FPSMeter:
    def __init__(self, alpha: float | None = None):
        self.alpha = float(config.FPS_ALPHA if alpha is None else alpha)
        self.last_ts = None
        self._ema_dt = None
        self.frames = 0
        self.fps = 0.0
        self.inst_fps = 0.0

    def tick(self):
        now = perf_counter()
        if self.last_ts is None:
            self.last_ts = now
            return
        dt = min(now - self.last_ts, config.FPS_MAX_DT_CLAMP)
        self.last_ts = now

        self.inst_fps = 1.0 / dt if dt > 0 else self.inst_fps
        self._ema_dt = dt if self._ema_dt is None else (1 - self.alpha) * self._ema_dt + self.alpha * dt
        self.fps = 1.0 / self._ema_dt if self._ema_dt and self._ema_dt > 0 else self.inst_fps
        self.frames += 1


def draw_fps_label(frame, meter: FPSMeter, x: int = 10, y: int = 30):
    _label(frame, f"{meter.fps:.{config.FPS_DECIMALS}f} FPS", x, y)
