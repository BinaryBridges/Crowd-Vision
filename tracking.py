from collections.abc import Sequence

import numpy as np

import config
from models import Status, TrackedFace
from utils import cosine_similarity, reduce_embedding_for_tracking


# ---------- Geometry ----------
def box_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    x1_1, y1_1, x2_1, y2_1 = b1
    x1_2, y1_2, x2_2, y2_2 = b2
    x1i, y1i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2i, y2i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    inter = float((x2i - x1i) * (y2i - y1i))
    a1 = float((x2_1 - x1_1) * (y2_1 - y1_1))
    a2 = float((x2_2 - x1_2) * (y2_2 - y1_2))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ---------- Small Utilities ----------
def _age_tracks(tracked: list[TrackedFace]) -> None:
    """Increment age, bump verification counters, and mark LOST if over max age."""
    for t in tracked:
        t.age += 1
        if t.status in {Status.CONFIRMED_KEY, Status.CONFIRMED_BAD}:
            t.frames_since_verification += 1
        if t.age > config.MAX_AGE_BEFORE_LOST:
            t.status = Status.LOST


def _valid_detections(
    detected_faces: Sequence,
) -> list[tuple[tuple[int, int, int, int], object]]:
    """
    Return (bbox, face_obj) pairs only for sufficiently large faces.
    Keeping bbox and face together avoids index drift after filtering.
    """
    out: list[tuple[tuple[int, int, int, int], object]] = []
    for f in detected_faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        if (x2 - x1) >= config.MIN_FACE_SIZE and (y2 - y1) >= config.MIN_FACE_SIZE:
            out.append(((x1, y1), (x2, y2), f))
        else:
            out.append(((x1, y1, x2, y2), f))
    # Normalize to (bbox, f)
    normed: list[tuple[tuple[int, int, int, int], object]] = []
    for item in out:
        if len(item[0]) == 2:
            (x1, y1), (x2, y2), f = item
            normed.append(((x1, y1, x2, y2), f))
        else:
            normed.append(item)
    return normed


def _category_to_status(category: str | None) -> Status:
    """Only 'bad' is disallowed; everything else is treated as allowed."""
    return Status.CONFIRMED_BAD if category == "bad" else Status.CONFIRMED_KEY


def _recognize(
    centroids: np.ndarray,
    names: np.ndarray,
    categories: np.ndarray,
    embedding: np.ndarray,
) -> tuple[str, float, str | None]:
    """Return (label, confidence, category) by cosine similarity to centroids."""
    if centroids.size == 0:
        return "Unknown", 0.0, None
    sims = cosine_similarity(centroids, embedding)
    idx = int(np.argmax(sims))
    best = float(sims[idx])
    if best >= config.SIM_THRESHOLD:
        return names[idx], best, (categories[idx] if len(categories) > idx else None)
    return "Unknown", 0.0, None


def _best_match_score(track: TrackedFace, bbox: tuple[int, int, int, int], emb: np.ndarray) -> float:
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
        emb_sim = float(cosine_similarity(track.embedding_tracking.reshape(1, -1), emb_reduced)[0])

    if track.status in {Status.CONFIRMED_KEY, Status.CONFIRMED_BAD}:
        # Confirmed: require strong embedding OR very large IoU; then combine
        return (
            (emb_sim * 0.7 + overlap * 0.3)
            if (emb_sim > config.CONF_STRONG_SIM or overlap > config.CONF_STRONG_SIM)
            else 0.0
        )
    # Tentative: prioritize IoU, but consider embedding a bit
    return (overlap * 0.7 + emb_sim * 0.3) if overlap > config.TENTATIVE_MIN_IOU else 0.0


def _confirm_if_ready(
    t: TrackedFace,
    known_centroids: np.ndarray,
    known_names: np.ndarray,
    known_categories: np.ndarray,
    notified: dict[str, int],
    frame_index: int,
) -> None:
    """Transition TENTATIVE -> CONFIRMED when hits meet threshold and handle notification."""
    if t.hit_count < config.MIN_HITS_TO_CONFIRM:
        return

    if t.label != "Unknown" and known_centroids.size > 0:
        idxs = np.where(known_names == t.label)[0]
        if len(idxs) > 0:
            cat = str(known_categories[idxs[0]])
            t.category = cat
            t.status = _category_to_status(cat)
            prefix = "🔑 KEY PERSON" if t.status == Status.CONFIRMED_KEY else "⚠️ BAD PERSON"
        else:
            t.category = None
            t.status = Status.CONFIRMED_KEY
            prefix = "✅ PERSON"
    else:
        t.category = None
        t.status = Status.CONFIRMED_KEY
        prefix = "✅ UNKNOWN"

    _notify_once(t, notified, frame_index, prefix=prefix)


def _notify_once(t: TrackedFace, notified: dict[str, int], frame_index: int, prefix: str) -> None:
    """Throttle notifications using NOTIFY_COOLDOWN_FRAMES."""
    key = f"{t.label}:{getattr(t, 'category', None)}:{t.status.name}"
    if key not in notified or (frame_index - notified[key]) >= config.NOTIFY_COOLDOWN_FRAMES:
        if t.label == "Unknown":
            print(f"[NOTIFICATION] {prefix} detected and confirmed")
        else:
            role_txt = f" [{getattr(t, 'category', None)}]" if getattr(t, "category", None) else ""
            print(f"[NOTIFICATION] {prefix} - {t.label}{role_txt} detected and confirmed")
        notified[key] = frame_index


def _update_tentative_track(
    t: TrackedFace,
    emb: np.ndarray,
    known_centroids: np.ndarray,
    known_names: np.ndarray,
    known_categories: np.ndarray,
    notified: dict[str, int],
    frame_index: int,
) -> None:
    """Update a TENTATIVE track with fresh recognition + drift check; confirm if ready."""
    new_label, new_conf, new_cat = _recognize(known_centroids, known_names, known_categories, emb)

    if t.label != new_label:
        print(f"[IDENTITY CHANGED] Face ID {t.face_id}: {t.label} -> {new_label} (confidence: {new_conf:.3f})")
        t.hit_count = 1
        t.label = new_label
        t.confidence = new_conf
        t.embedding_full = emb
        t.embedding_tracking = reduce_embedding_for_tracking(emb)
        t.category = new_cat if new_label != "Unknown" else None
    else:
        # Check drift on reduced embeddings
        if t.embedding_tracking is not None:
            emb_reduced = reduce_embedding_for_tracking(emb)
            sim = float(cosine_similarity(t.embedding_tracking.reshape(1, -1), emb_reduced)[0])
            if sim < config.TRACK_DRIFT_MIN_SIM:
                # Reset tentative if drifted too far
                print(f"[DRIFT] Face ID {t.face_id} drifted (sim={sim:.3f}) -> resetting tentative hits")
                t.hit_count = 1
                t.embedding_tracking = emb_reduced
        else:
            t.embedding_tracking = reduce_embedding_for_tracking(emb)

        t.confidence = max(t.confidence, new_conf)
        if new_label != "Unknown":
            t.category = new_cat

        # Accumulate evidence
        t.hit_count += 1

    # Confirm if enough evidence accumulated
    _confirm_if_ready(t, known_centroids, known_names, known_categories, notified, frame_index)


def _update_confirmed_track(
    t: TrackedFace,
    emb: np.ndarray,
    known_centroids: np.ndarray,
    known_names: np.ndarray,
    known_categories: np.ndarray,
) -> None:
    """Periodic verification for confirmed tracks; demote on mismatch, refresh on match."""
    print(f"[PERIODIC CHECK] Verifying Face ID {t.face_id} ({t.label}) after {t.frames_since_verification} frames")
    v_label, v_conf, v_cat = _recognize(known_centroids, known_names, known_categories, emb)

    if t.label != v_label:
        print(f"[IDENTITY MISMATCH] Face ID {t.face_id}: {t.label} -> {v_label} - demoting to tentative")
        t.status = Status.TENTATIVE
        t.hit_count = 1
        t.label = v_label
        t.confidence = v_conf
        t.embedding_full = emb
        t.embedding_tracking = reduce_embedding_for_tracking(emb)
        t.frames_since_verification = 0
        t.category = v_cat if v_label != "Unknown" else None
        return

    # Same identity: update status if category says so and refresh embeddings
    if v_label != "Unknown":
        t.category = v_cat
        if v_cat is not None:
            new_status = _category_to_status(str(v_cat))
            if new_status != t.status:
                t.status = new_status
                print(f"[STATUS UPDATE] Face ID {t.face_id} ({t.label}) -> {t.status.name}")

    print(f"[VERIFIED] Face ID {t.face_id} ({t.label}) stays confirmed; refreshing embeddings")
    t.embedding_full = emb
    t.embedding_tracking = reduce_embedding_for_tracking(emb)
    t.frames_since_verification = 0


def _reactivate_or_create_track(
    tracked: list[TrackedFace],
    bbox: tuple[int, int, int, int],
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
        r_label, r_conf, r_cat = _recognize(known_centroids, known_names, known_categories, emb)
        best_track.bbox = bbox
        best_track.age = 0
        best_track.hit_count = 1
        best_track.status = Status.TENTATIVE
        best_track.label = r_label
        best_track.confidence = r_conf
        best_track.embedding_full = emb
        best_track.embedding_tracking = reduce_embedding_for_tracking(emb)
        best_track.frames_since_verification = 0
        best_track.category = r_cat if r_label != "Unknown" else None
        print(f"[REACTIVATED] Face ID {best_track.face_id} as '{best_track.label}' (sim={best_sim:.3f})")
        return face_id_counter

    # New track
    label, conf, cat = _recognize(known_centroids, known_names, known_categories, emb)
    new_face = TrackedFace(
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
    new_face.category = cat if label != "Unknown" else None
    tracked.append(new_face)
    print(f"[NEW FACE] ID {face_id_counter} ({label}) - confidence: {conf:.3f}")
    return face_id_counter + 1


def _enforce_one_box_per_identity(tracked: list[TrackedFace]) -> None:
    """Keep the strongest track per name (confirmed > confidence > hits); mark others LOST."""
    by_name: dict[str, list[TrackedFace]] = {}
    for t in tracked:
        if t.status != Status.LOST:
            by_name.setdefault(t.label, []).append(t)

    for faces in by_name.values():
        if len(faces) <= 1:
            continue
        # Rank: confirmed > higher confidence > more hits > younger age (prefer fresh)
        faces.sort(
            key=lambda x: (
                x.status in {Status.CONFIRMED_KEY, Status.CONFIRMED_BAD},
                x.confidence,
                x.hit_count,
                -x.age,
            ),
            reverse=True,
        )
        best = faces[0]
        for other in faces[1:]:
            if other is best:
                continue
            other.status = Status.LOST
            print(f"[DEDUP] Keeping Face ID {best.face_id} for '{best.label}', dropping Face ID {other.face_id}")


def _purge_stale_lost(tracked: list[TrackedFace]) -> list[TrackedFace]:
    """Remove tracks that have been LOST for more than 2× max age."""
    return [t for t in tracked if not (t.status == Status.LOST and t.age > config.MAX_AGE_BEFORE_LOST * 2)]


# ---------- Main Tracking ----------
def track_and_update_faces(
    tracked: list[TrackedFace],
    detected_faces: Sequence,  # from InsightFace app.get(frame)
    known_names: np.ndarray,
    known_centroids: np.ndarray,
    known_categories: np.ndarray,
    face_id_counter: int,
    notified: dict[str, int],
    frame_index: int,
) -> tuple[list[TrackedFace], int]:
    """
    Associate detections with tracks, update state, reactivate/create tracks,
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
            if score > best_score:
                best_score, best_track = score, t

        if best_track is None or best_score < config.ASSOC_MIN_SCORE:
            continue  # will try to reactivate/create later

        # Claim the detection for this track
        matched_det_idxs.add(det_idx)
        used_track_ids.add(best_track.face_id)

        # Update track geometry & age reset
        best_track.bbox = bbox
        best_track.age = 0

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
        elif best_track.status in {Status.CONFIRMED_KEY, Status.CONFIRMED_BAD}:
            # periodic re-verify
            if best_track.frames_since_verification >= config.PERIODIC_VERIFY_EVERY:
                _update_confirmed_track(
                    best_track,
                    emb,
                    known_centroids,
                    known_names,
                    known_categories,
                )
            else:
                # Light refresh
                best_track.embedding_tracking = reduce_embedding_for_tracking(emb)
        # LOST is handled by reactivation path

    # 4) For any unassociated detections: try to reactivate or create tracks
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
