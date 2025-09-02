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

from insightface.app import FaceAnalysis

import config
from tracking import track_and_update_faces
from utils import FPSMeter, draw_fps_label, draw_tracked_faces


def main():
    print(config.BANNER)
    print(f"\n=== {config.APP_TITLE}: Face Tracking\n")

    # Initialize detector/embedding
    app = FaceAnalysis(
        name=config.DETECTION_MODEL_NAME,
        allowed_modules=["detection", "recognition", "genderage"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)

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
        fps_meter = FPSMeter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            faces = app.get(frame)
            tracked, face_id_counter = track_and_update_faces(
                tracked=tracked,
                detected_faces=faces,
                face_id_counter=face_id_counter,
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
