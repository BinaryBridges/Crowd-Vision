import contextlib
from typing import Any

import cv2
import numpy as np

from app import config


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def reduce_embedding_for_tracking(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    k = min(config.TRACK_VERIFY_VECTOR_SIZE, emb.shape[-1])
    return _l2(emb[..., :k])


def _short_gender(g) -> str | None:
    if isinstance(g, (int, np.integer)):
        return "M" if int(g) == 1 else "F"
    if isinstance(g, str):
        s = g.lower()
        if s.startswith("m"):
            return "M"
        if s.startswith("f"):
            return "F"
    return None


def format_demographics(face_obj) -> dict[str, Any]:
    """Return {'gender': 'M'|'F'|None, 'age': int|None} for a face object."""
    g = getattr(face_obj, "gender", None)
    if g is None:
        g = getattr(face_obj, "sex", None)
    gender = _short_gender(g)

    age_val = getattr(face_obj, "age", None)
    age_out = None
    if age_val is not None:
        with contextlib.suppress(Exception):
            age_out = round(float(age_val))

    return {"gender": gender, "age": age_out}


def load_bgr_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img
