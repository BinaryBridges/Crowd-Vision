from dataclasses import dataclass
from enum import Enum

import numpy as np


class Status(Enum):
    """Tracking status for a face"""

    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3


BBox = tuple[int, int, int, int]


@dataclass
class TrackedFace:
    """A face being tracked across frames."""

    face_id: int
    bbox: BBox
    label: str
    status: Status
    hit_count: int = 0
    age: int = 0
    embedding_full: np.ndarray | None = None
    embedding_tracking: np.ndarray | None = None
    frames_since_verification: int = 0
    age_years: float | None = None
    gender: str | None = None
