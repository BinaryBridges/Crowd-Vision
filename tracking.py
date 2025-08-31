from collections.abc import Sequence
from typing import Dict, List, Tuple, Optional
import numpy as np

import config
from models import Status, TrackedFace
from utils import cosine_similarity, reduce_embedding_for_tracking


# ---------- Geometry ----------
def box_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1_1, y1_1, x2_1, y2_1 = b1
    x1_2, y1_2, x2_2, y2_2 = b2
    x1i, y1i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2i, y2i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    inter = (x2i - x1i) * (y2i - y1i)
    a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ---------- Small Utilities ----------
def _age_tracks(tracked: List[TrackedFace]) -> None:
    """Increment age, bump verification counters, and mark LOST if over max age."""
    for t in tracked:
        t.age += 1
        if t.status in (Status.CONFIRMED_KEY, Status.CONFIRMED_BAD):
            t.frames_since_verification += 1
        if t.age > config.MAX_AGE_BEFORE_LOST:
            t.status = Status.LOST


def _valid_detections(
    detected_faces: Sequence,
) -> List[Tuple[Tuple[int, int, int, int], object]]:
    """
    Return (bbox, face_obj) pairs only for sufficiently large faces.
    Keeping bbox and face together avoids index drift after filtering.
    """
    out: List[Tuple[Tuple[int, int, int, int], object]] = []
    for f in detected_faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        if (x2 - x1) >= config.MIN_FACE_SIZE and (y2 - y1) >= config.MIN_FACE_SIZE:
            out.append(((x1, y1, x2, y2), f))
    return out


def _category_to_status(category: Optional[str]) -> Status:
    """Only 'bad' is disallowed; everything else is treated as allowed."""
    return Status.CONFIRMED_BAD if category == "bad" else Status.CONFIRMED_KEY


def _recognize(
    centroids: np.ndarray,
    names: np.ndarray,
    categories: np.ndarray,
    embedding: np.ndarray,
) -> Tuple[str, float, Optional[str]]:
    """Return (label, confidence, category) by cosine similarity to centroids."""
    if centroids.size == 0:
        return "Unknown", 0.0, None
    sims = cosine_similarity(centroids, embedding)
    idx = int(np.argmax(sims))
    best = float(sims[idx])
    if best >= config.SIM_THRESHOLD:
        return names[idx], best, categories[idx]
    return "Unknown", 0.0, None


def _best_match_score(
    track: TrackedFace, bbox: Tuple[int, int, int, int], emb: np.ndarray
) -> float:
    """
    Combined IoU + (reduced) embedding consistency score with status-aware rules.
    Uses config thresholds; returns 0.0 if below acceptance criteria.
    """
    if track.status == Status.LOST:
        return 0.0

    overlap = box_iou(bbox, track.bbox)
    emb_sim = 0.0
    if track.embedding_tracking is not None:
        emb_reduced = reduce_embedding_for_tracking(emb)
        emb_sim = float(
            cosine_similarity(track.embedding_tracking.reshape(1, -1), emb_reduced)[0]
        )

    if track.status in (Status.CONFIRMED_KEY, Status.CONFIRMED_BAD):
        # Confirmed: require strong embedding OR very large IoU; then combine
        return (
            (emb_sim * 0.7 + overlap * 0.3)
            if (emb_sim > config.CONF_STRONG_SIM or overlap > config.CONF_STRONG_SIM)
            else 0.0
        )
    else:
        # Tentative: prioritize IoU, but consider embedding a bit
        return (
            (overlap * 0.7 + emb_sim * 0.3)
            if overlap > config.TENTATIVE_MIN_IOU
            else 0.0
        )


def _confirm_if_ready(
    t: TrackedFace,
    known_centroids: np.ndarray,
    known_names: np.ndarray,
    known_categories: np.ndarray,
    notified: Dict[str, int],
    frame_index: int,
) -> None:
    """Transition TENTATIVE -> CONFIRMED when hits meet threshold and handle notification."""
    if t.hit_count < config.MIN_HITS_TO_CONFIRM:
        return

    if t.label != "Unknown" and known_centroids.size > 0:
        # Look up category for this label if present
        idxs = np.where(known_names == t.label)[0]
        if len(idxs) > 0:
            cat = known_categories[idxs[0]]
            t.status = _category_to_status(str(cat))
            prefix = (
                "🔑 KEY PERSON" if t.status == Status.CONFIRMED_KEY else "⚠️ BAD PERSON"
            )
        else:
            print(f"[WARNING] Person '{t.label}' recognized but not found in arrays.")
            t.status = Status.TENTATIVE
            prefix = "❓ UNVERIFIED"
    else:
        t.status = Status.CONFIRMED_KEY
        prefix = "❓ UNKNOWN PERSON"

    t.frames_since_verification = 0
    status_text = "CONFIRMED_UNKNOWN" if t.label == "Unknown" else t.status.name
    print(f"[{status_text}] Face ID {t.face_id} ({t.label}) after {t.hit_count} hits")

    key = t.label
    if (
        key not in notified
        or (frame_index - notified[key]) >= config.NOTIFY_COOLDOWN_FRAMES
    ):
        if t.label == "Unknown":
            print(f"[NOTIFICATION] {prefix} detected and confirmed")
        else:
            print(f"[NOTIFICATION] {prefix} - {t.label} detected and confirmed")
        notified[key] = frame_index


def _update_tentative_track(
    t: TrackedFace,
    emb: np.ndarray,
    known_centroids: np.ndarray,
    known_names: np.ndarray,
    known_categories: np.ndarray,
    notified: Dict[str, int],
    frame_index: int,
) -> None:
    """Update a TENTATIVE track with fresh recognition + drift check; confirm if ready."""
    new_label, new_conf, _ = _recognize(
        known_centroids, known_names, known_categories, emb
    )

    if t.label != new_label:
        print(
            f"[IDENTITY CHANGED] Face ID {t.face_id}: {t.label} -> {new_label} (confidence: {new_conf:.3f})"
        )
        t.hit_count = 1
        t.label = new_label
        t.confidence = new_conf
        t.embedding_full = emb
        t.embedding_tracking = reduce_embedding_for_tracking(emb)
    else:
        # Check drift on reduced embeddings
        if t.embedding_tracking is not None:
            reduced_now = reduce_embedding_for_tracking(emb)
            sim = float(
                cosine_similarity(t.embedding_tracking.reshape(1, -1), reduced_now)[0]
            )
            if sim < config.TRACK_DRIFT_MIN_SIM:
                print(
                    f"[EMBEDDING DRIFT] Face ID {t.face_id} ({t.label}): tracking similarity {sim:.3f} - resetting"
                )
                t.hit_count = 1
        t.label = new_label
        t.confidence = new_conf
        t.embedding_full = emb
        t.embedding_tracking = reduce_embedding_for_tracking(emb)

    _confirm_if_ready(
        t, known_centroids, known_names, known_categories, notified, frame_index
    )


def _update_confirmed_track(
    t: TrackedFace,
    emb: np.ndarray,
    known_centroids: np.ndarray,
    known_names: np.ndarray,
    known_categories: np.ndarray,
) -> None:
    """Periodic verification for confirmed tracks; demote on mismatch, refresh on match."""
    print(
        f"[PERIODIC CHECK] Verifying Face ID {t.face_id} ({t.label}) after {t.frames_since_verification} frames"
    )
    v_label, v_conf, v_cat = _recognize(
        known_centroids, known_names, known_categories, emb
    )

    if t.label != v_label:
        print(
            f"[IDENTITY MISMATCH] Face ID {t.face_id}: {t.label} -> {v_label} - demoting to tentative"
        )
        t.status = Status.TENTATIVE
        t.hit_count = 1
        t.label = v_label
        t.confidence = v_conf
        t.embedding_full = emb
        t.embedding_tracking = reduce_embedding_for_tracking(
            emb
        )  # ensure tracking vector is refreshed
        t.frames_since_verification = 0
        return

    # Same identity: update status if category says so and refresh embeddings
    if v_label != "Unknown" and v_cat is not None:
        new_status = _category_to_status(str(v_cat))
        if new_status != t.status:
            t.status = new_status
            print(f"[STATUS UPDATE] Face ID {t.face_id} ({t.label}) -> {t.status.name}")

    print(f"[VERIFIED] Face ID {t.face_id} ({t.label}) - confidence: {v_conf:.3f}")
    t.confidence = v_conf
    t.embedding_full = emb
    t.embedding_tracking = reduce_embedding_for_tracking(emb)
    t.frames_since_verification = 0


def _reactivate_or_create_track(
    tracked: List[TrackedFace],
    bbox: Tuple[int, int, int, int],
    emb: np.ndarray,
    known_centroids: np.ndarray,
    known_names: np.ndarray,
    known_categories: np.ndarray,
    face_id_counter: int,
) -> int:
    """Try to re-attach to a LOST track using full embedding; otherwise create a new track."""
    best_track = None
    best_sim = 0.0

    for t in tracked:
        if t.embedding_full is None:
            continue
        sim = float(cosine_similarity(t.embedding_full.reshape(1, -1), emb)[0])
        if sim > best_sim and sim > config.REACTIVATE_MIN_SIM:
            best_sim = sim
            best_track = t

    if best_track and best_track.status == Status.LOST:
        # Reactivate
        best_track.bbox = bbox
        best_track.age = 0
        best_track.hit_count = 1
        best_track.status = Status.TENTATIVE
        best_track.embedding_full = emb
        best_track.embedding_tracking = reduce_embedding_for_tracking(emb)
        best_track.frames_since_verification = 0

        label, conf, _ = _recognize(known_centroids, known_names, known_categories, emb)
        best_track.label = label
        best_track.confidence = conf
        print(
            f"[REACTIVATED] Face ID {best_track.face_id} ({label}) - similarity: {best_sim:.3f}"
        )
        return face_id_counter

    if best_track:
        print(
            f"[DUPLICATE PREVENTED] Skipping detection - already tracked as ID {best_track.face_id} ({best_track.label}) - similarity: {best_sim:.3f}"
        )
        return face_id_counter

    # New track
    label, conf, _ = _recognize(known_centroids, known_names, known_categories, emb)
    tracked.append(
        TrackedFace(
            face_id=face_id_counter,
            bbox=bbox,
            label=label,
            confidence=conf,
            status=Status.TENTATIVE,
            hit_count=1,
            age=0,
            embedding_full=emb,
            embedding_tracking=reduce_embedding_for_tracking(emb),
            frames_since_verification=0,
        )
    )
    print(f"[NEW FACE] ID {face_id_counter} ({label}) - confidence: {conf:.3f}")
    return face_id_counter + 1


def _enforce_one_box_per_identity(tracked: List[TrackedFace]) -> None:
    """Keep the strongest track per name (confirmed > confidence > hits); mark others LOST."""
    by_name: Dict[str, List[TrackedFace]] = {}
    for t in tracked:
        if t.status != Status.LOST:
            by_name.setdefault(t.label, []).append(t)

    for faces in by_name.values():
        if len(faces) <= 1:
            continue
        faces.sort(
            key=lambda f: (
                f.status not in (Status.CONFIRMED_KEY, Status.CONFIRMED_BAD),
                -f.confidence,
                -f.hit_count,
            )
        )
        keep = faces[0]
        for f in faces[1:]:
            f.status = Status.LOST
            f.age = config.MAX_AGE_BEFORE_LOST + 1
            print(
                f"[DUPLICATE REMOVED] Face ID {f.face_id} ({f.label}) removed - keeping ID {keep.face_id}"
            )


def _purge_stale_lost(tracked: List[TrackedFace]) -> List[TrackedFace]:
    """Remove tracks that have been LOST for more than 2× max age."""
    return [
        t
        for t in tracked
        if not (t.status == Status.LOST and t.age > config.MAX_AGE_BEFORE_LOST * 2)
    ]


# ---------- Main Tracking ----------
def track_and_update_faces(
    tracked: List[TrackedFace],
    detected_faces: Sequence,  # from InsightFace app.get(frame)
    known_names: np.ndarray,
    known_centroids: np.ndarray,
    known_categories: np.ndarray,
    face_id_counter: int,
    notified: Dict[str, int],
    frame_index: int,
) -> Tuple[List[TrackedFace], int]:
    """Associate detections with tracks, update state, reactivate/create tracks,
    dedupe identities, and purge stale entries. Public API unchanged.
    """
    # 1) Age existing tracks
    _age_tracks(tracked)

    # 2) Filter detections and keep (bbox, face_obj) paired
    det_items = _valid_detections(detected_faces)

    matched_det_idxs, used_track_ids = set(), set()

    # 3) Associate each detection to the best track (greedy, status-aware scoring)
    for det_idx, (bbox, df) in enumerate(det_items):
        emb = df.normed_embedding.astype(np.float32)

        best_track = None
        best_score = 0.0
        for t in tracked:
            if t.face_id in used_track_ids or t.status == Status.LOST:
                continue
            score = _best_match_score(t, bbox, emb)
            if score > best_score and score > config.ASSOC_MIN_SCORE:
                best_score = score
                best_track = t

        if best_track is None:
            continue

        # Update matched track with this observation
        best_track.bbox = bbox
        best_track.age = 0
        best_track.hit_count += 1

        if best_track.status == Status.TENTATIVE:
            _update_tentative_track(
                best_track,
                emb,
                known_centroids,
                known_names,
                known_categories,
                notified,
                frame_index,
            )
        elif best_track.frames_since_verification >= config.PERIODIC_VERIFY_EVERY:
            _update_confirmed_track(
                best_track, emb, known_centroids, known_names, known_categories
            )

        matched_det_idxs.add(det_idx)
        used_track_ids.add(best_track.face_id)

    # 4) Handle unmatched detections: reactivate a LOST track or create a new one
    for det_idx, (bbox, df) in enumerate(det_items):
        if det_idx in matched_det_idxs:
            continue
        emb = df.normed_embedding.astype(np.float32)
        face_id_counter = _reactivate_or_create_track(
            tracked,
            bbox,
            emb,
            known_centroids,
            known_names,
            known_categories,
            face_id_counter,
        )

    # 5) Enforce one box per identity and purge stale LOST tracks
    _enforce_one_box_per_identity(tracked)
    tracked = _purge_stale_lost(tracked)

    return tracked, face_id_counter
