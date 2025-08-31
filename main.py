import time

import cv2
from insightface.app import FaceAnalysis
import numpy as np

import config
from data_loader import load_reference_identities
from tracking import track_and_update_faces
from utils import draw_text_label, draw_tracked_faces, prune_old_notifications


def main():
    print(config.BANNER)
    print("\nInitializing face analysis model...")
    app = FaceAnalysis(name=config.DETECTION_MODEL_NAME)
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)
    print("Model initialized.\n")

    # Load known identities
    key_people, bad_people = load_reference_identities(app, config.KNOWN_DIR)
    all_people = key_people + bad_people
    known_names = np.array([p.name for p in all_people], dtype=object)
    known_categories = np.array(
        (["key"] * len(key_people)) + (["bad"] * len(bad_people)), dtype=object
    )
    known_centroids = (
        np.stack([p.centroid for p in all_people]).astype(np.float32)
        if all_people
        else np.empty((0, config.DETECTION_VECTOR_SIZE), np.float32)
    )

    print(f"[DATA STRUCTURES] Loaded {len(key_people)} key, {len(bad_people)} bad")
    print(
        f"[DATA STRUCTURES] Names: {len(known_names)}, Categories: {len(known_categories)}, Centroids: {known_centroids.shape[0]}"
    )
    print(f"[DEBUG] Key: {[p.name for p in key_people]}")
    print(f"[DEBUG] Bad: {[p.name for p in bad_people]}")
    print(f"[DEBUG] Names array: {list(known_names)}")
    print(f"[DEBUG] Categories array: {list(known_categories)}\n")

    # Video capture
    print("Initializing webcam...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {config.CAMERA_INDEX}).")
    print("Done.\nPress 'q' to quit.\n")

    # Tracking state
    fps, prev_time = 0.0, time.time()
    face_id_counter = 0
    tracked = []
    notified = {}
    frame_idx = 0

    # Loop
    while True:
        frame_idx += 1
        ok, frame = cap.read()
        if not ok:
            break

        faces = app.get(frame)
        tracked, face_id_counter = track_and_update_faces(
            tracked,
            faces,
            known_names,
            known_centroids,
            known_categories,
            face_id_counter,
            notified,
            frame_idx,
        )

        if frame_idx % 100 == 0:
            prune_old_notifications(notified, frame_idx)

        draw_tracked_faces(frame, tracked)

        if config.SHOW_FPS:
            now = time.time()
            inst = 1.0 / max(1e-6, (now - prev_time))
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            prev_time = now
            draw_text_label(frame, f"{fps:.1f} FPS", 10, 30)

        cv2.imshow(config.WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
