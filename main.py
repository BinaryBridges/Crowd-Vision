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
from utils import FPSMeter, draw_detections, draw_fps_label


def main():
    print(config.BANNER)
    print(f"{config.APP_TITLE}\n")
    app = FaceAnalysis(
        name=config.DETECTION_MODEL_NAME,
        allowed_modules=["detection", "genderage"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)

    # Video capture
    print("Initializing webcam...")
    cap = None
    try:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {config.CAMERA_INDEX}).")
        print("Done.\nPress 'q' to quit.\n")

        fps_meter = FPSMeter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            faces = app.get(frame)
            draw_detections(frame, faces)

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
