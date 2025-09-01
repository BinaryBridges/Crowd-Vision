import os
import warnings
from contextlib import suppress

os.environ["INSIGHTFACE_ONNX_PROVIDERS"] = "CPUExecutionProvider"
os.environ["ORT_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings(
    "ignore",
    message="Specified provider 'CUDAExecutionProvider' is not in available provider names.*",
    category=UserWarning,
)

import cv2

cv2.setNumThreads(1)

import numpy as np
from insightface.app import FaceAnalysis

import config
from data_loader import load_reference_identities
from tracking import track_and_update_faces
from utils import FPSMeter, draw_fps_label, draw_tracked_faces, expire_notifications


def main():
    print(config.BANNER)
    print("\nInitializing face analysis model...")
    app = FaceAnalysis(
        name=config.DETECTION_MODEL_NAME,
        allowed_modules=["detection", "recognition"],  # skip age/gender/3D landmarks
        providers=["CPUExecutionProvider"],  # pin provider explicitly
    )
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)
    print("Model initialized.\n")

    people, categories = load_reference_identities(app, config.KNOWN_DIR)
    known_names = np.array([p.name for p in people], dtype=object)
    known_categories = np.array(categories, dtype=object)
    known_centroids = (
        np.stack([p.centroid for p in people]).astype(np.float32)
        if people
        else np.empty((0, config.DETECTION_VECTOR_SIZE), np.float32)
    )

    # Diagnostics
    cat_counts = {}
    for c in known_categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    cats_summary = ", ".join(f"{c}:{n}" for c, n in sorted(cat_counts.items())) or "none"
    print(f"[DATA STRUCTURES] Loaded people: {len(people)} ({cats_summary})")
    print(
        f"[DATA STRUCTURES] Names: {len(known_names)}, Categories: {len(known_categories)}, "
        f"Centroids: {known_centroids.shape[0]}"
    )
    print(f"[DEBUG] Names array: {list(known_names)}")
    print(f"[DEBUG] Categories array: {list(known_categories)}\n")

    # Video capture
    print("Initializing webcam...")
    cap = None
    try:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {config.CAMERA_INDEX}).")
        print("Done.\nPress 'q' to quit.\n")

        face_id_counter = 0
        tracked = []
        notified = {}
        frame_idx = 0
        fps_meter = FPSMeter()

        while True:
            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break

            faces = app.get(frame)
            expire_notifications(notified, frame_idx)

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

            draw_tracked_faces(frame, tracked)

            fps_meter.tick()
            if config.SHOW_FPS:
                draw_fps_label(frame, fps_meter, 10, 30)

            cv2.imshow(config.WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if cap is not None:
            with suppress(cv2.error):
                cap.release()
        with suppress(cv2.error):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
