
import os

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
FACE_IMAGES_DIR = os.path.join(DATA_DIR, "face_images")
DB_PATH         = os.path.join(DATA_DIR, "faces.db")
EXPORTS_DIR     = os.path.join(BASE_DIR, "logs", "exports")

# ─── CAMERA SETTINGS ──────────────────────────────────────────────────────────
# 0 = default webcam, 1 = USB camera
# For IP camera: "rtsp://username:password@ip_address:port/stream"
CAMERA_SOURCE = 0
CAMERA_NAME   = "Main Camera"

# Process every Nth frame (1 = all frames, 2 = every other, etc.)
# Higher = faster but slightly less responsive
FRAME_SKIP = 3

# ─── DEEPFACE RECOGNITION SETTINGS ───────────────────────────────────────────
# Cosine similarity threshold (0.0 to 1.0)
# Higher = stricter match required
# ArcFace recommended range: 0.65 – 0.72
RECOGNITION_TOLERANCE = 0.68

# Minimum face height in pixels — ignore tiny/far-away faces
MIN_FACE_SIZE = 60

# LEARNING NOTE:
#   DETECTION_MODEL is no longer needed — DeepFace handles detection internally.
#   We kept RECOGNITION_TOLERANCE as the main tuning knob.

# ─── DUPLICATE LOG PREVENTION ────────────────────────────────────────────────
# Don't log the same person again within this many seconds
DUPLICATE_LOG_INTERVAL_SECONDS = 60

# ─── ADMIN CREDENTIALS ────────────────────────────────────────────────────────
# IMPORTANT: Change these before deploying to any real environment!
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ─── UI SETTINGS ──────────────────────────────────────────────────────────────
APP_TITLE = "Smart Face Recognition System"
APP_ICON  = "🎯"

# Bounding box colors (BGR format for OpenCV)
COLOR_KNOWN   = (0, 255, 0)      # Green for recognized persons
COLOR_UNKNOWN = (0, 0, 255)      # Red for unknown persons
COLOR_TEXT    = (255, 255, 255)  # White text labels

# ─── AUTO-CREATE DIRECTORIES ──────────────────────────────────────────────────
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
