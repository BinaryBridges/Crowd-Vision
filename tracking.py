import operator
from collections.abc import Sequence

import numpy as np

import config
from models import Status, TrackedFace
from utils import cosine_similarity, reduce_embedding_for_tracking

_DET_NMS_IOU = 0.55
_MATCH_GROW_PX = 14
_SPAWN_SUPPRESS_IOU = 0.70
_SPAWN_SUPPRESS_SIM = 0.70
_SUPPRESS_IOU = 0.85


# --- geometry ---
def _iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b1
    x1b, y1b, x2b, y2b = b2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = float((xi2 - xi1) * (yi2 - yi1))
    a1 = float((x2 - x1) * (y2 - y1))
    a2 = float((x2b - x1b) * (y2b - y1b))
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


def _expand(b: tuple[int, int, int, int], px: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    return (x1 - px, y1 - px, x2 + px, y2 + px)


# --- utilities ---
def _age_tracks(tracked: list[TrackedFace]) -> None:
    for t in tracked:
        t.age += 1
        if t.status == Status.CONFIRMED:
            t.frames_since_verification += 1
        if t.age > config.MAX_AGE_BEFORE_LOST:
            t.status = Status.LOST


def _valid_detections(detected_faces: Sequence) -> list[tuple[tuple[int, int, int, int], object]]:
    out: list[tuple[tuple[int, int, int, int], object]] = []
    for f in detected_faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        if (x2 - x1) >= config.MIN_FACE_SIZE and (y2 - y1) >= config.MIN_FACE_SIZE:
            out.append(((x1, y1, x2, y2), f))
    return out


def _nms(det_items: list[tuple[tuple[int, int, int, int], object]]):
    if not det_items:
        return det_items
    scored = [(float(getattr(o, "det_score", 0.5)), b, o) for b, o in det_items]
    scored.sort(key=operator.itemgetter(0), reverse=True)
    kept, kept_boxes = [], []
    for _, b, o in scored:
        if all(_iou(b, kb) < _DET_NMS_IOU for kb in kept_boxes):
            kept.append((b, o))
            kept_boxes.append(b)
    return kept


def _embeddings(face_obj) -> tuple[np.ndarray | None, np.ndarray | None]:
    emb = getattr(face_obj, "normed_embedding", None)
    if emb is None:
        emb = getattr(face_obj, "embedding", None)
        if emb is None:
            return None, None
        emb = np.asarray(emb, dtype=np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
    else:
        emb = np.asarray(emb, dtype=np.float32)
    return emb, reduce_embedding_for_tracking(emb) if emb is not None else None


def _assoc_score(track: TrackedFace, bbox: tuple[int, int, int, int], emb_red: np.ndarray | None) -> float:
    if track.status == Status.LOST:
        return 0.0
    overlap = _iou(bbox, _expand(track.bbox, _MATCH_GROW_PX))
    emb_sim = 0.0
    if emb_red is not None and track.embedding_tracking is not None:
        emb_sim = float(cosine_similarity(track.embedding_tracking.reshape(1, -1), emb_red)[0])
    if track.status == Status.CONFIRMED:
        return (
            emb_sim * 0.7 + overlap * 0.3
            if (emb_sim > config.CONF_STRONG_SIM or overlap > config.CONF_STRONG_SIM)
            else 0.0
        )
    return (overlap * 0.7 + emb_sim * 0.3) if overlap > config.TENTATIVE_MIN_IOU else 0.0


def _confirm_if_ready(t: TrackedFace) -> None:
    if t.hit_count >= config.MIN_HITS_TO_CONFIRM and t.status == Status.TENTATIVE:
        t.status = Status.CONFIRMED


def _maybe_merge_with_active(
    tracked: list[TrackedFace],
    bbox: tuple[int, int, int, int],
    emb_full: np.ndarray | None,
    emb_red: np.ndarray | None,
) -> bool:
    best, best_score = None, 0.0
    for t in tracked:
        if t.status == Status.LOST or t.age == 0:
            continue
        overlap = _iou(bbox, _expand(t.bbox, _MATCH_GROW_PX))
        sim = (
            float(cosine_similarity(t.embedding_tracking.reshape(1, -1), emb_red)[0])
            if (emb_red is not None and t.embedding_tracking is not None)
            else 0.0
        )
        if overlap >= _SPAWN_SUPPRESS_IOU or sim >= _SPAWN_SUPPRESS_SIM:
            score = sim * 0.7 + overlap * 0.3
            if score > best_score:
                best, best_score = t, score
    if best is None:
        return False
    best.bbox = bbox
    best.age = 0
    if emb_full is not None:
        best.embedding_full = emb_full
    if emb_red is not None:
        best.embedding_tracking = emb_red
    if best.status == Status.TENTATIVE:
        best.hit_count += 1
        _confirm_if_ready(best)
    else:
        best.frames_since_verification = 0
    return True


def _reactivate_or_create_track(
    tracked: list[TrackedFace],
    bbox: tuple[int, int, int, int],
    emb_full: np.ndarray | None,
    emb_red: np.ndarray | None,
    face_id_counter: int,
) -> int:
    best, best_sim = None, 0.0
    if emb_full is not None:
        for t in tracked:
            if t.status != Status.LOST or t.embedding_full is None:
                continue
            sim = float(cosine_similarity(t.embedding_full.reshape(1, -1), emb_full)[0])
            if sim > best_sim and sim > config.REACTIVATE_MIN_SIM:
                best, best_sim = t, sim
    if best is not None:
        best.bbox = bbox
        best.age = 0
        best.hit_count = 1
        best.status = Status.TENTATIVE
        best.label = "Unknown"
        if emb_full is not None:
            best.embedding_full = emb_full
        if emb_red is not None:
            best.embedding_tracking = emb_red
        best.frames_since_verification = 0
        return face_id_counter

    if _maybe_merge_with_active(tracked, bbox, emb_full, emb_red):
        return face_id_counter

    for t in tracked:
        if t.status == Status.LOST or t.age != 0:
            continue
        overlap = _iou(bbox, _expand(t.bbox, _MATCH_GROW_PX))
        sim = (
            float(cosine_similarity(t.embedding_tracking.reshape(1, -1), emb_red)[0])
            if (emb_red is not None and t.embedding_tracking is not None)
            else 0.0
        )
        if overlap >= _SPAWN_SUPPRESS_IOU or sim >= _SPAWN_SUPPRESS_SIM:
            return face_id_counter

    tracked.append(
        TrackedFace(
            face_id=face_id_counter,
            bbox=bbox,
            label="Unknown",
            status=Status.TENTATIVE,
            hit_count=1,
            age=0,
            embedding_full=emb_full if emb_full is not None else None,
            embedding_tracking=emb_red if emb_red is not None else None,
            frames_since_verification=0,
        )
    )
    return face_id_counter + 1


def _dedupe_active(tracked: list[TrackedFace]) -> None:
    active = [i for i, t in enumerate(tracked) if t.status != Status.LOST and t.age == 0]
    to_drop = set()
    for i in range(len(active)):
        ti = tracked[active[i]]
        for j in range(i + 1, len(active)):
            tj = tracked[active[j]]
            if _iou(ti.bbox, tj.bbox) >= _SUPPRESS_IOU:
                keep_i = (ti.hit_count > tj.hit_count) or (ti.hit_count == tj.hit_count and ti.face_id < tj.face_id)
                drop = tj if keep_i else ti
                to_drop.add(drop.face_id)
    if to_drop:
        for t in tracked:
            if t.face_id in to_drop:
                t.status = Status.LOST


def _purge(tracked: list[TrackedFace]) -> list[TrackedFace]:
    return [t for t in tracked if not (t.status == Status.LOST and t.age > config.MAX_AGE_BEFORE_LOST * 2)]


def track_and_update_faces(
    tracked: list[TrackedFace],
    detected_faces: Sequence,
    face_id_counter: int,
) -> tuple[list[TrackedFace], int]:
    _age_tracks(tracked)

    det_items = _nms(_valid_detections(detected_faces))
    matched_det_idxs, used_track_ids = set(), set()

    for det_idx, (bbox, df) in enumerate(det_items):
        emb_full, emb_red = _embeddings(df)

        best_track, best_score = None, 0.0
        for t in tracked:
            if t.face_id in used_track_ids or t.status == Status.LOST:
                continue
            score = _assoc_score(t, bbox, emb_red)
            if score > best_score:
                best_track, best_score = t, score

        if best_track is None or best_score < config.ASSOC_MIN_SCORE:
            continue

        matched_det_idxs.add(det_idx)
        used_track_ids.add(best_track.face_id)
        best_track.bbox = bbox
        best_track.age = 0
        if emb_full is not None:
            best_track.embedding_full = emb_full
        if emb_red is not None:
            best_track.embedding_tracking = emb_red
        if best_track.status == Status.TENTATIVE:
            best_track.hit_count += 1
            _confirm_if_ready(best_track)
        else:
            best_track.frames_since_verification = 0

    for det_idx, (bbox, df) in enumerate(det_items):
        if det_idx in matched_det_idxs:
            continue
        emb_full, emb_red = _embeddings(df)
        face_id_counter = _reactivate_or_create_track(tracked, bbox, emb_full, emb_red, face_id_counter)

    _dedupe_active(tracked)
    tracked = _purge(tracked)

    return tracked, face_id_counter
