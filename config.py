# ---- Data / Model ----
KNOWN_DIR = "known_faces"  # Root folder holding category/name subfolders
DETECTION_MODEL_NAME = "buffalo_l"  # InsightFace model suite name
CAMERA_INDEX = 0  # Default webcam index
DET_SIZE = (640, 640)  # Detector input size

# ---- Categories ----
# Add a new category by adding a string here (e.g., "vip"). Non-"bad" categories
# will be treated as "allowed" (green) by default.
PERSON_CATEGORIES = ["key", "bad"]

# ---- Thresholds & Hyperparameters ----
SIM_THRESHOLD = 0.38  # Cosine similarity threshold for recognition (0-1)
MIN_FACE_SIZE = 32  # Minimum face size to consider (pixels)
MIN_HITS_TO_CONFIRM = 10  # Frames needed to confirm a tentative face
MAX_AGE_BEFORE_LOST = 5  # Frames a face can be unseen before marked LOST
PERIODIC_VERIFY_EVERY = 10  # Verify confirmed faces every N frames
NOTIFY_COOLDOWN_FRAMES = 100  # Frames to wait before re-notifying same person

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
