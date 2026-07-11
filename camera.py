
import cv2
import threading
import time
from datetime import datetime

from config import (
    CAMERA_SOURCE, CAMERA_NAME, FRAME_SKIP,
    DUPLICATE_LOG_INTERVAL_SECONDS
)
from database import (
    load_all_face_encodings,
    add_recognition_log,
    get_recent_log_time,
    add_pending_approval,
)
from face_utils import save_snapshot, recognize_faces_in_frame


class FaceRecognitionCamera:
    """
    Manages the live camera feed and face recognition loop.

    Usage:
        cam = FaceRecognitionCamera()
        cam.start()
        # ... get frames via cam.get_current_frame()
        cam.stop()
    """

    def __init__(self, source=None, camera_name=None):
        self.source = source if source is not None else CAMERA_SOURCE
        self.camera_name = camera_name or CAMERA_NAME

        # Thread-safe state
        self._lock = threading.Lock()
        self._running = False
        self._current_frame = None
        self._current_results = []
        self._thread = None

        # Statistics
        self.fps = 0
        self.frame_count = 0
        self.last_error = None

        # Face data (reloaded from DB on start)
        self.known_encodings = []
        self.known_metadata = []

        # Duplicate log prevention: {person_id: datetime}
        self._last_logged = {}

    # ── Public Interface ──────────────────────────────────────────────────────

    def start(self):
        """Start the recognition thread."""
        if self._running:
            return
        self._reload_face_data()
        self._running = True
        self._thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] Started — source: {self.source}")

    def stop(self):
        """Stop the recognition thread gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("[Camera] Stopped.")

    def is_running(self):
        return self._running

    def get_current_frame(self):
        """Get the latest annotated frame (thread-safe)."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def get_current_results(self):
        """Get the latest recognition results (thread-safe)."""
        with self._lock:
            return list(self._current_results)

    def reload_faces(self):
        """Reload face data from DB (call after registering a new person)."""
        self._reload_face_data()
        print(f"[Camera] Reloaded {len(self.known_encodings)} face(s).")

    # ── Internal Methods ──────────────────────────────────────────────────────

    def _reload_face_data(self):
        """Load all face encodings from the database."""
        self.known_encodings, self.known_metadata = load_all_face_encodings()
        print(f"[Camera] Loaded {len(self.known_encodings)} registered face(s).")

    def _recognition_loop(self):
        """
        Main loop: read frames → detect + recognize faces → save logs.
        Runs in a background thread.
        """
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self.last_error = f"Cannot open camera source: {self.source}"
            print(f"[Camera] ERROR: {self.last_error}")
            self._running = False
            return

        # Optimize camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_idx = 0
        fps_start = time.time()

        while self._running:
            ret, frame = cap.read()

            if not ret:
                # Camera disconnected or stream ended
                self.last_error = "Camera feed lost. Retrying..."
                print(f"[Camera] Feed lost. Retrying in 2s...")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(self.source)
                continue

            self.last_error = None
            frame_idx += 1

            # Skip frames for performance (process every Nth frame)
            if frame_idx % FRAME_SKIP == 0:
                results, annotated = recognize_faces_in_frame(
                    frame,
                    self.known_encodings,
                    self.known_metadata
                )

                # Process recognition results
                for result in results:
                    self._handle_recognition(result, frame)

                with self._lock:
                    self._current_frame = annotated
                    self._current_results = results
            else:
                # Non-processed frame: show raw so feed stays smooth
                with self._lock:
                    if self._current_frame is None:
                        self._current_frame = frame

            # FPS calculation
            self.frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self.fps = round(self.frame_count / elapsed, 1)
                self.frame_count = 0
                fps_start = time.time()

        cap.release()

    def _handle_recognition(self, result, frame):
        """
        Called when a face is recognized (known or unknown).
        - Known person → log the attendance
        - Unknown person → save snapshot + add to pending approvals for admin review
        """
        person_id = result["person_id"]
        now = datetime.now()

        # Duplicate prevention — don't log same person twice within interval
        last_time = self._last_logged.get(person_id)
        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < DUPLICATE_LOG_INTERVAL_SECONDS:
                return

        snapshot_path = ""

        if result["status"] == "Unknown":
            # Save face snapshot
            try:
                top, right, bottom, left = result["location"]
                face_crop = frame[top:bottom, left:right]
                if face_crop.size > 0:
                    snapshot_path = save_snapshot(face_crop, label="unknown")

                    # Save to pending approvals for admin review
                    if result.get("embedding") is not None:
                        add_pending_approval(
                            snapshot_path=snapshot_path,
                            face_encoding=result["embedding"],
                            camera_name=self.camera_name
                        )
            except Exception as e:
                print(f"[Camera] Could not save pending approval: {e}")

        # Always log the recognition event
        add_recognition_log(
            person_id=result["person_id"],
            person_name=result["name"],
            department=result.get("department", ""),
            camera_name=self.camera_name,
            status=result["status"],
            confidence=result.get("confidence", 0.0),
            snapshot_path=snapshot_path,
        )

        self._last_logged[person_id] = now
        print(f"[Camera] Logged: {result['name']} ({result['status']}) at {now.strftime('%H:%M:%S')}")


# ─── Standalone Test ──────────────────────────────────────────────────────────
# Run this file directly to test the camera without the dashboard:
#   python camera.py

if __name__ == "__main__":
    from database import initialize_database, seed_admin
    from config import ADMIN_USERNAME, ADMIN_PASSWORD

    initialize_database()
    seed_admin(ADMIN_USERNAME, ADMIN_PASSWORD)

    cam = FaceRecognitionCamera()
    cam.start()

    print("Press Q to quit.")
    while True:
        frame = cam.get_current_frame()
        if frame is not None:
            cv2.imshow("Face Recognition - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()