from collections.abc import Sequence

import numpy as np

import config
from models import Status, TrackedFace
from utils import cosine_similarity, reduce_embedding_for_tracking

# Geometry Helper


def iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
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


# Helpers


def _age_and_decay(tracked: list[TrackedFace]) -> None:
    for t in tracked:
        t.age += 1
        if t.status in (Status.CONFIRMED_KEY, Status.CONFIRMED_BAD):
            t.frames_since_verification += 1
        if t.age > config.MAX_AGE_BEFORE_LOST:
            t.status = Status.LOST


def _filter_valid_bboxes(detected_faces: Sequence) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for f in detected_faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        if (x2 - x1) >= config.MIN_FACE_SIZE and (y2 - y1) >= config.MIN_FACE_SIZE:
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def _category_to_status(category: str) -> Status:
    # Only "bad" is treated as disallowed; all others = allowed
    return Status.CONFIRMED_BAD if category == "bad" else Status.CONFIRMED_KEY


def _recognize(
    centroids: np.ndarray,
    names: np.ndarray,
    categories: np.ndarray,
    embedding: np.ndarray,
):
    """Return (label, confidence, category) using cosine similarity to centroids."""
    if centroids.size == 0:
        return "Unknown", 0.0, None
    sims = cosine_similarity(centroids, embedding)
    idx = int(np.argmax(sims))
    best = float(sims[idx])
    if best >= config.SIM_THRESHOLD:
        return names[idx], best, categories[idx]
    return "Unknown", 0.0, None


def _enforce_one_box_per_identity(tracked: list[TrackedFace]) -> None:
    by_name: dict[str, list[TrackedFace]] = {}
    for t in tracked:
        if t.status != Status.LOST:
            by_name.setdefault(t.label, []).append(t)

    for name, faces in by_name.items():
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


def _purge_stale_lost(tracked: list[TrackedFace]) -> list[TrackedFace]:
    return [
        t
        for t in tracked
        if not (t.status == Status.LOST and t.age > config.MAX_AGE_BEFORE_LOST * 2)
    ]


# Main Tracking Function


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
    """Match detections to existing tracks (IoU + embedding consistency),
    manage tentative/confirmed statuses, reactivate lost, add new tracks,
    and keep only one box per identity.
    """
    # 1) Age existing tracks
    _age_and_decay(tracked)

    # 2) Prepare detection bboxes (skip tiny faces)
    det_bboxes = _filter_valid_bboxes(detected_faces)
    matched_det_idxs, matched_track_ids = set(), set()

    # 3) Attempt association between detections and tracks
    for det_idx, bbox in enumerate(det_bboxes):
        df = detected_faces[det_idx]
        emb = df.normed_embedding.astype(np.float32)

        best_track = None
        best_score = 0.0

        for t in tracked:
            if t.status == Status.LOST or t.face_id in matched_track_ids:
                continue

            overlap = iou(bbox, t.bbox)

            # For confirmed tracks, we prefer embedding consistency if necessary; else IoU can be enough
            emb_sim = 0.0
            if t.embedding_tracking is not None:
                emb_reduced = reduce_embedding_for_tracking(emb)
                emb_sim = cosine_similarity(
                    t.embedding_tracking.reshape(1, -1), emb_reduced
                )[0]

            if t.status in (Status.CONFIRMED_KEY, Status.CONFIRMED_BAD):
                score = (
                    emb_sim * 0.7 + overlap * 0.3
                    if (emb_sim > 0.7 or overlap > 0.7)
                    else 0.0
                )
            else:
                score = overlap * 0.7 + emb_sim * 0.3 if overlap > 0.3 else 0.0

            if score > best_score and score > 0.4:
                best_score = score
                best_track = t

        if best_track is None:
            continue

        # 4) Update matched track with new observation
        best_track.bbox = bbox
        best_track.age = 0
        best_track.hit_count += 1

        if best_track.status == Status.TENTATIVE:
            # Re-identify against all known identities each time to stabilize early
            new_label, new_conf, new_cat = _recognize(
                known_centroids, known_names, known_categories, emb
            )

            if best_track.label != new_label:
                print(
                    f"[IDENTITY CHANGED] Face ID {best_track.face_id}: {best_track.label} -> {new_label} (confidence: {new_conf:.3f})"
                )
                best_track.hit_count = 1
                best_track.label = new_label
                best_track.confidence = new_conf
                best_track.embedding_full = emb
                best_track.embedding_tracking = reduce_embedding_for_tracking(emb)
            else:
                # Same identity: check tracking embedding drift
                if best_track.embedding_tracking is not None:
                    reduced_now = reduce_embedding_for_tracking(emb)
                    sim = cosine_similarity(
                        best_track.embedding_tracking.reshape(1, -1), reduced_now
                    )[0]
                    if sim < 0.6:
                        print(
                            f"[EMBEDDING DRIFT] Face ID {best_track.face_id} ({best_track.label}): tracking similarity {sim:.3f} - resetting"
                        )
                        best_track.hit_count = 1
                best_track.label = new_label
                best_track.confidence = new_conf
                best_track.embedding_full = emb
                best_track.embedding_tracking = reduce_embedding_for_tracking(emb)

            # Confirm once stable enough
            if best_track.hit_count >= config.MIN_HITS_TO_CONFIRM:
                if best_track.label != "Unknown" and known_centroids.size > 0:
                    # Derive category and map to a status
                    idxs = np.where(known_names == best_track.label)[0]
                    if len(idxs) > 0:
                        cat = known_categories[idxs[0]]
                        best_track.status = _category_to_status(str(cat))
                        prefix = (
                            "🔑 KEY PERSON"
                            if best_track.status == Status.CONFIRMED_KEY
                            else "⚠️ BAD PERSON"
                        )
                    else:
                        print(
                            f"[WARNING] Person '{best_track.label}' recognized but not found in arrays."
                        )
                        best_track.status = Status.TENTATIVE
                        prefix = "❓ UNVERIFIED"
                else:
                    best_track.status = Status.CONFIRMED_KEY
                    prefix = "❓ UNKNOWN PERSON"

                best_track.frames_since_verification = 0
                status_text = (
                    "CONFIRMED_UNKNOWN"
                    if best_track.label == "Unknown"
                    else best_track.status.name
                )
                print(
                    f"[{status_text}] Face ID {best_track.face_id} ({best_track.label}) after {best_track.hit_count} hits"
                )

                # Notification (cooldown-gated)
                key = best_track.label
                if (
                    key not in notified
                    or (frame_index - notified[key]) >= config.NOTIFY_COOLDOWN_FRAMES
                ):
                    if best_track.label == "Unknown":
                        print(f"[NOTIFICATION] {prefix} detected and confirmed")
                    else:
                        print(
                            f"[NOTIFICATION] {prefix} - {best_track.label} detected and confirmed"
                        )
                    notified[key] = frame_index

        # Confirmed track: periodic verification only
        elif best_track.frames_since_verification >= config.PERIODIC_VERIFY_EVERY:
            print(
                f"[PERIODIC CHECK] Verifying Face ID {best_track.face_id} ({best_track.label}) after {best_track.frames_since_verification} frames"
            )
            v_label, v_conf, v_cat = _recognize(
                known_centroids, known_names, known_categories, emb
            )
            if best_track.label != v_label:
                print(
                    f"[IDENTITY MISMATCH] Face ID {best_track.face_id}: {best_track.label} -> {v_label} - demoting to tentative"
                )
                best_track.status = Status.TENTATIVE
                best_track.hit_count = 1
                best_track.label = v_label
                best_track.confidence = v_conf
                best_track.embedding_full = emb
                best_track.frames_since_verification = 0
            else:
                # Update status if category changed (unlikely) and refresh embeddings
                if v_label != "Unknown" and v_cat is not None:
                    new_status = _category_to_status(str(v_cat))
                    if new_status != best_track.status:
                        best_track.status = new_status
                        print(
                            f"[STATUS UPDATE] Face ID {best_track.face_id} ({best_track.label}) -> {best_track.status.name}"
                        )
                print(
                    f"[VERIFIED] Face ID {best_track.face_id} ({best_track.label}) - confidence: {v_conf:.3f}"
                )
                best_track.confidence = v_conf
                best_track.embedding_full = emb
                best_track.embedding_tracking = reduce_embedding_for_tracking(emb)
                best_track.frames_since_verification = 0

        matched_det_idxs.add(det_idx)
        matched_track_ids.add(best_track.face_id)

    # 5) Handle unmatched detections: re-activate or create new tracks
    for det_idx, bbox in enumerate(det_bboxes):
        if det_idx in matched_det_idxs:
            continue
        df = detected_faces[det_idx]
        emb = df.normed_embedding.astype(np.float32)

        # Try to match via full embedding with any existing track (incl. LOST)
        best_track = None
        best_sim = 0.0
        for t in tracked:
            if t.embedding_full is not None:
                sim = cosine_similarity(t.embedding_full.reshape(1, -1), emb)[0]
                if sim > best_sim and sim > 0.75:
                    best_sim = sim
                    best_track = t

        if best_track and best_track.status == Status.LOST:
            # Reactivate a previously lost track
            best_track.bbox = bbox
            best_track.age = 0
            best_track.hit_count = 1
            best_track.status = Status.TENTATIVE
            best_track.embedding_full = emb
            best_track.embedding_tracking = reduce_embedding_for_tracking(emb)
            best_track.frames_since_verification = 0

            # Update recognition
            label, conf, _ = _recognize(
                known_centroids, known_names, known_categories, emb
            )
            best_track.label = label
            best_track.confidence = conf
            print(
                f"[REACTIVATED] Face ID {best_track.face_id} ({label}) - similarity: {best_sim:.3f}"
            )
        elif best_track:
            print(
                f"[DUPLICATE PREVENTED] Skipping detection - already tracked as ID {best_track.face_id} ({best_track.label}) - similarity: {best_sim:.3f}"
            )
            continue
        else:
            # Brand new track
            label, conf, _ = _recognize(
                known_centroids, known_names, known_categories, emb
            )
            new_track = TrackedFace(
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
            tracked.append(new_track)
            print(f"[NEW FACE] ID {face_id_counter} ({label}) - confidence: {conf:.3f}")
            face_id_counter += 1

    # 6) Keep one box per identity; purge stale lost tracks
    _enforce_one_box_per_identity(tracked)
    tracked = _purge_stale_lost(tracked)

    return tracked, face_id_counter
