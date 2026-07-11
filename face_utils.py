"""
face_utils.py - Core face processing utilities (DeepFace version)

LEARNING NOTE:
  DeepFace works differently from the old face_recognition library:

  OLD WAY (face_recognition / dlib):
    - Converts face → 128 numbers → compare numbers
    - One fixed model, decent accuracy

  NEW WAY (DeepFace):
    - Uses deep neural networks (like ArcFace, VGG-Face, Facenet)
    - Converts face → 512 numbers (more detail = better accuracy)
    - We use ArcFace model — currently one of the best in the world
    - Also handles face DETECTION separately using RetinaFace or OpenCV

  FLOW:
    1. DETECT  → Find face location in image (bounding box)
    2. EMBED   → Convert face region to a 512-number vector (embedding)
    3. COMPARE → Cosine similarity between two vectors (1.0 = identical)
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

# DeepFace imports
from deepface import DeepFace
from deepface.modules import detection as deepface_detection

from config import (
    RECOGNITION_TOLERANCE, MIN_FACE_SIZE,
    FACE_IMAGES_DIR, COLOR_KNOWN, COLOR_UNKNOWN, COLOR_TEXT
)

# ─── Model Configuration ──────────────────────────────────────────────────────
# ArcFace = best accuracy for real-world use
# Other options: "VGG-Face", "Facenet", "Facenet512", "SFace"
EMBEDDING_MODEL = "ArcFace"

# Face detector: "opencv" = fastest, "retinaface" = most accurate
# Use "opencv" for live camera (speed), "retinaface" for registration (accuracy)
DETECTOR_BACKEND = "opencv"

# Cosine similarity threshold (0.0 to 1.0)
# Higher = stricter. ArcFace works well at 0.68
COSINE_THRESHOLD = 0.68


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODING  (called during registration)
# ═══════════════════════════════════════════════════════════════════════════════

def get_face_encoding_from_image(image_path):
    """
    Load an image file and extract the face embedding using DeepFace + ArcFace.

    Returns:
        (embedding: numpy array of 512 floats, message: str)
        or (None, error_message) if failed

    LEARNING NOTE:
        DeepFace.represent() does two things in one call:
          1. Detects the face location
          2. Runs the neural network to get the embedding
        It returns a list — one dict per face found.
        Each dict has: {"embedding": [...], "facial_area": {...}}
    """
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=EMBEDDING_MODEL,
            detector_backend="retinaface",  # More accurate for registration
            enforce_detection=True,          # Raise error if no face found
            align=True,                      # Align face for better accuracy
        )

        if len(results) == 0:
            return None, "No face detected. Please use a clear front-facing photo."

        if len(results) > 1:
            return None, "Multiple faces detected. Please upload a photo with only one person."

        embedding = np.array(results[0]["embedding"])
        return embedding, "Success"

    except ValueError as e:
        # DeepFace raises ValueError when no face is found
        return None, "No face detected in the image. Please use a clear, well-lit front-facing photo."
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def get_face_encoding_from_frame(frame):
    """
    Extract face embedding from a live OpenCV frame (for camera registration).

    Returns (embedding, message).
    """
    try:
        # DeepFace accepts BGR frames directly (handles conversion internally)
        results = DeepFace.represent(
            img_path=frame,
            model_name=EMBEDDING_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )

        if not results:
            return None, "No face detected in frame. Please look directly at the camera."

        if len(results) > 1:
            return None, "Multiple faces detected. Please ensure only one person is in frame."

        embedding = np.array(results[0]["embedding"])
        return embedding, "Success"

    except ValueError:
        return None, "No face detected. Please look directly at the camera with good lighting."
    except Exception as e:
        return None, f"Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# RECOGNITION  (called every frame during live camera)
# ═══════════════════════════════════════════════════════════════════════════════

def recognize_faces_in_frame(frame, known_encodings, known_metadata, scale=0.5):
    """
    Process one video frame: detect all faces and recognize each one.

    Args:
        frame:           OpenCV BGR image
        known_encodings: list of numpy arrays (embeddings from DB)
        known_metadata:  list of dicts with person info
        scale:           resize factor for speed (0.5 = half size)

    Returns:
        results:          list of recognition result dicts
        annotated_frame:  frame with boxes and labels drawn

    LEARNING NOTE:
        For LIVE camera we use a two-step approach:
          Step 1 → Detect faces with OpenCV (fast, runs every frame)
          Step 2 → Get embedding and compare (slightly slower)

        We resize DOWN for detection speed, then scale coordinates back UP.
    """
    results = []

    # Step 1: Resize for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    try:
        # Step 2: Detect faces + get embeddings in one DeepFace call
        df_results = DeepFace.represent(
            img_path=small_frame,
            model_name=EMBEDDING_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,   # Don't crash if no face found
            align=True,
        )
    except Exception:
        df_results = []

    for face_data in df_results:
        embedding = np.array(face_data["embedding"])
        area = face_data.get("facial_area", {})

        # Get face location from DeepFace result
        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 0)
        h = area.get("h", 0)

        # Filter tiny/unclear faces
        if h < MIN_FACE_SIZE * scale:
            continue

        # Scale coordinates back to original frame size
        left   = int(x / scale)
        top    = int(y / scale)
        right  = int((x + w) / scale)
        bottom = int((y + h) / scale)

        # Step 3: Compare embedding against all known faces
        identity = _match_embedding(embedding, known_encodings, known_metadata)
        identity["location"] = (top, right, bottom, left)
        results.append(identity)

    # Step 4: Draw boxes and labels
    annotated_frame = _draw_annotations(frame.copy(), results)

    return results, annotated_frame

def _match_embedding(embedding, known_encodings, known_metadata):
    """
    Compare a face embedding against all registered faces.
    """
    # Always return the embedding so pending approvals can store it
    unknown_result = {
        "person_id":  "UNKNOWN",
        "name":       "Unknown Person",
        "department": "",
        "status":     "Unknown",
        "confidence": 0.0,
        "embedding":  embedding,  # ← always pass embedding back
    }

    if not known_encodings:
        return unknown_result

    # Compute cosine similarity with all known faces
    similarities = []
    for known_enc in known_encodings:
        similarity = _cosine_similarity(embedding, known_enc)
        similarities.append(similarity)

    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    if best_score >= COSINE_THRESHOLD:
        confidence = round(best_score * 100, 1)
        person = known_metadata[best_idx]
        return {
            "person_id":  person["person_id"],
            "name":       person["name"],
            "department": person["department"],
            "status":     "Known",
            "confidence": confidence,
            "embedding":  None,
        }

    # Below threshold — unknown but embedding is stored for admin approval
    unknown_result["embedding"] = embedding
    return unknown_result


def _cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    Result: 0.0 (different) to 1.0 (identical)

    Formula: (A · B) / (|A| × |B|)
    """
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    dot_product = np.dot(a, b)
    magnitude   = np.linalg.norm(a) * np.linalg.norm(b)

    if magnitude == 0:
        return 0.0

    return float(dot_product / magnitude)


# ═══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_annotations(frame, results):
    """Draw bounding boxes and name labels on the video frame."""
    for result in results:
        top, right, bottom, left = result["location"]
        is_known = result["status"] == "Known"
        color = COLOR_KNOWN if is_known else COLOR_UNKNOWN

        # Bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label text
        label = result["name"]
        if is_known:
            label += f"  {result['confidence']}%"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame,
                      (left, bottom - 30),
                      (left + label_size[0] + 10, bottom),
                      color, cv2.FILLED)

        cv2.putText(frame, label,
                    (left + 5, bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLOR_TEXT, 1, cv2.LINE_AA)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def save_face_image(image_source, person_id, is_file=True):
    """
    Save a face image to the face_images directory.

    Args:
        image_source: file path (str) or numpy array (frame)
        person_id:    used to name the file
        is_file:      True if image_source is a path, False if it's a frame
    """
    filename = f"{person_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    save_path = os.path.join(FACE_IMAGES_DIR, filename)

    try:
        if is_file:
            import shutil
            shutil.copy(image_source, save_path)
        else:
            cv2.imwrite(save_path, image_source)
        return save_path
    except Exception as e:
        print(f"[face_utils] Could not save image: {e}")
        return ""


def save_snapshot(frame, label="unknown"):
    """Save a snapshot of an unknown person's face."""
    snapshots_dir = os.path.join(FACE_IMAGES_DIR, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(snapshots_dir, f"{label}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def validate_image_has_face(image_path):
    """Quick check: does this image contain exactly one clear face?"""
    encoding, message = get_face_encoding_from_image(image_path)
    return encoding is not None, message