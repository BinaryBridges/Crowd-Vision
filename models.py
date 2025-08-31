from dataclasses import dataclass
from enum import Enum

import numpy as np


class Status(Enum):
    """Tracking status for a face."""

    CONFIRMED_KEY = 1  # Confirmed allowed person (or unknown confirmed)
    CONFIRMED_BAD = 2  # Confirmed disallowed person
    TENTATIVE = 3  # Detected but not yet confirmed
    LOST = 4  # No longer tracked (aged out)


@dataclass
class Person:
    """Known identity built from reference images."""

    name: str
    centroid: np.ndarray  # L2-normalized 512-d embedding
    num_refs: int
    category: str = "key"


@dataclass
class TrackedFace:
    """A face being tracked across frames."""

    face_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    label: str
    confidence: float
    status: Status
    hit_count: int = 0
    age: int = 0
    embedding_full: np.ndarray | None = None
    embedding_tracking: np.ndarray | None = None
    frames_since_verification: int = 0
