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


def draw_text_label(
    img: np.ndarray, text: str, x: int, y: int, scale: float = 0.6, thickness: int = 2
):
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
        return (
            (0, 200, 0) if face.label != "Unknown" else (255, 0, 0)
        )  # green or blue for unknown confirmed
    if face.status == Status.CONFIRMED_BAD:
        return (0, 0, 255)  # red
    if face.status == Status.TENTATIVE:
        return (0, 255, 255)  # yellow
    return (128, 128, 128)  # gray


def draw_tracked_faces(frame: np.ndarray, tracked: list[TrackedFace]):
    for face in tracked:
        if face.status == Status.LOST:
            continue
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_for_face_status(face), 2)
        status_text = face.status.name.lower()
        label = (
            f"{face.label} {face.confidence:.2f} ({status_text})"
            if face.confidence > 0
            else f"{face.label} ({status_text})"
        )
        draw_text_label(frame, label, x1, y1)


def prune_old_notifications(notified: dict[str, int], current_frame: int):
    expired = [
        name
        for name, frm in notified.items()
        if current_frame - frm > config.NOTIFY_COOLDOWN_FRAMES
    ]
    for name in expired:
        notified.pop(name, None)
