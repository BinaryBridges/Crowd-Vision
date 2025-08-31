# ---- Data / Model ----
KNOWN_DIR = "known_faces"  # Root folder holding category/name subfolders
DETECTION_MODEL_NAME = "buffalo_s"  # InsightFace model suite name
CAMERA_INDEX = 0  # Default webcam index
DET_SIZE = (320, 320)  # Detector input size

# ---- Categories ----
# Add a new category by adding a string here (e.g., "vip"). Non-"bad" categories
# will be treated as "allowed" (green) by default.
PERSON_CATEGORIES = ["key", "bad"]

# ---- Thresholds & Hyperparameters ----
SIM_THRESHOLD = 0.38  # Cosine similarity threshold for recognition (0-1)
MIN_FACE_SIZE = 32  # Minimum face size to consider (pixels)
MIN_HITS_TO_CONFIRM = 10  # Frames needed to confirm a tentative face
MAX_AGE_BEFORE_LOST = 5  # Frames unseen before marking LOST
PERIODIC_VERIFY_EVERY = 10  # Verify confirmed faces every N frames
NOTIFY_COOLDOWN_FRAMES = 100  # Frames to wait before re-notifying same person

# Association/verification fine-tuning (moved from magic numbers in code)
ASSOC_MIN_SCORE = 0.4  # Minimum association score to pair a detection with a track
CONF_STRONG_SIM = 0.7  # "Strong" sim/IoU threshold for confirmed tracks
TENTATIVE_MIN_IOU = 0.3  # Minimum IoU to consider a tentative match
TRACK_DRIFT_MIN_SIM = 0.6  # Reduced-embedding similarity below which we reset hits (drift)
REACTIVATE_MIN_SIM = 0.75  # Full-embedding similarity to reactivate a LOST track

# ---- Embedding settings ----
DETECTION_VECTOR_SIZE = 512  # ArcFace embedding size
TRACK_VERIFY_VECTOR_SIZE = 128  # Reduced embedding used in tracking verification

# ---- Display ----
SHOW_FPS = True
WINDOW_TITLE = "Face ID (webcam)"
BANNER = (
    "██████╗░░█████╗░██████╗░░█████╗░██╗░░██╗  ███████╗██╗░░░██╗███████╗\n"
    "██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║░░██║  ██╔════╝╚██╗░██╔╝██╔════╝\n"
    "██████╔╝██║░░██║██████╔╝██║░░╚═╝███████║  █████╗░░░╚████╔╝░█████╗░░\n"
    "██╔═══╝░██║░░██║██╔══██╗██║░░██╗██╔══██║  ██╔══╝░░░░╚██╔╝░░██╔══╝░░\n"
    "██║░░░░░╚█████╔╝██║░░██║╚█████╔╝██║░░██║  ███████╗░░░██║░░░███████╗\n"
    "╚═╝░░░░░░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝  ╚══════╝░░░╚═╝░░░╚══════╝"
)

# --- FPS display tuning ---
FPS_ALPHA = 0.12  # EMA smoothing factor (0..1), larger = more responsive
FPS_MAX_DT_CLAMP = 0.5  # clamp a single frame's dt to avoid giant spikes
FPS_DECIMALS = 1  # decimals to show in the FPS label
