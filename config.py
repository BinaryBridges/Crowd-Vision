# App Branding
APP_TITLE = "CROWD VISION"
WINDOW_TITLE = f"{APP_TITLE} (webcam)"
BANNER = r"""
################################################################################
#                                                                              #
#                                C R O W D   V I S I O N                       #
#                                                                              #
################################################################################
"""

# Runtime / Model
CAMERA_INDEX = 0
DETECTION_MODEL_NAME = "buffalo_s"
DET_SIZE = (640, 640)

# Tracking thresholds
MIN_FACE_SIZE = 40  # pixels: ignore tiny detections
MAX_AGE_BEFORE_LOST = 20  # frames until a track becomes LOST if not matched
MIN_HITS_TO_CONFIRM = 3  # frames required to promote TENTATIVE -> CONFIRMED

# Association
ASSOC_MIN_SIM = 0.30  # minimal cosine similarity to accept association
CONF_STRONG_SIM = 0.70  # "strong" similarity gate for confirmed tracks
REACTIVATE_MIN_SIM = 0.75  # full-embedding cosine sim to revive LOST tracks

# Pixel-distance gates
MAX_CENTER_DIST_PX = 120  # reject association if centers farther than this
DET_NMS_DIST_PX = 24  # distance-based NMS for detector outputs
SPAWN_SUPPRESS_DIST_PX = 48  # prevent duplicate spawns near active tracks
DEDUPE_DIST_PX = 42  # drop one of two active tracks if too close

# Embedding settings
TRACK_VERIFY_VECTOR_SIZE = 128  # reduced dim used during tracking

# Display
SHOW_FPS = True
FPS_ALPHA = 0.12  # EMA smoothing factor for FPS
FPS_MAX_DT_CLAMP = 0.50  # clamp dt spikes in seconds
FPS_DECIMALS = 1  # FPS decimals

# Drawing
COLOR_UNKNOWN = (50, 180, 255)  # BGR for Unknown boxes/labels
TEXT_COLOR = (255, 255, 255)  # label text color
TEXT_BG = (30, 30, 30)  # label background
