# ---- App Branding ----
APP_TITLE = "CROWD VISION"
WINDOW_TITLE = f"{APP_TITLE} (webcam)"
BANNER = r"""
################################################################################
#                                                                              #
#                                C R O W D   V I S I O N                       #
#                                                                              #
################################################################################
"""

# ---- Runtime / Model ----
CAMERA_INDEX = 0
DETECTION_MODEL_NAME = "buffalo_s"  # small & fast bundle (SCRFD + ArcFace)
DET_SIZE = (320, 320)  # detector input size (w, h)

# ---- Tracking thresholds ----
MIN_FACE_SIZE = 40  # pixels: ignore tiny detections
MAX_AGE_BEFORE_LOST = 20  # frames until a track becomes LOST if not matched
MIN_HITS_TO_CONFIRM = 3  # frames required to promote TENTATIVE -> CONFIRMED
TENTATIVE_MIN_IOU = 0.30  # minimal IoU to consider tentative match
CONF_STRONG_SIM = 0.70  # "strong" sim/IoU gate for confirmed tracks
ASSOC_MIN_SCORE = 0.30  # minimal greedy association score to accept match
REACTIVATE_MIN_SIM = 0.75  # full-embedding cosine sim to revive LOST tracks

# ---- Embedding settings (ArcFace) ----
TRACK_VERIFY_VECTOR_SIZE = 128  # reduced dim used during tracking

# ---- Display ----
SHOW_FPS = True
FPS_ALPHA = 0.12  # EMA smoothing factor for FPS
FPS_MAX_DT_CLAMP = 0.50  # clamp dt spikes in seconds
FPS_DECIMALS = 1  # FPS decimals

# ---- Drawing ----
COLOR_UNKNOWN = (50, 180, 255)  # BGR for Unknown boxes/labels
TEXT_COLOR = (255, 255, 255)  # label text color
TEXT_BG = (30, 30, 30)  # label background
