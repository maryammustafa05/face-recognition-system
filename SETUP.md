# 📖 Smart Face Recognition System — Setup Guide & User Manual

## ──────────────────────────────────────────────────
## SYSTEM REQUIREMENTS
## ──────────────────────────────────────────────────

| Requirement | Minimum |
|-------------|---------|
| Python      | 3.8 – 3.11 |
| RAM         | 4 GB |
| Camera      | Any webcam / USB / IP camera |
| OS          | Windows 10+, Ubuntu 20.04+, macOS 12+ |

---

## ──────────────────────────────────────────────────
## STEP 1: INSTALL DEPENDENCIES
## ──────────────────────────────────────────────────

### On Windows

```bash
# 1. Install Visual C++ Build Tools (required for dlib)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 2. Install cmake
pip install cmake

# 3. Install dlib (face recognition core)
pip install dlib

# 4. Install all other requirements
pip install -r requirements.txt
```

### On Ubuntu / Debian Linux

```bash
# Install system dependencies first
sudo apt update
sudo apt install -y python3-pip cmake libopenblas-dev liblapack-dev libx11-dev

# Install Python packages
pip install -r requirements.txt
```

### On macOS

```bash
# Install Homebrew first (if not installed): https://brew.sh
brew install cmake

# Install Python packages
pip install -r requirements.txt
```

---

## ──────────────────────────────────────────────────
## STEP 2: CONFIGURE THE SYSTEM
## ──────────────────────────────────────────────────

Open `config.py` and adjust:

```python
CAMERA_SOURCE = 0           # 0 = default webcam, 1 = second camera
ADMIN_PASSWORD = "admin123" # CHANGE THIS before deploying!
RECOGNITION_TOLERANCE = 0.50  # 0.45 (strict) to 0.55 (lenient)
DUPLICATE_LOG_INTERVAL_SECONDS = 30  # Don't re-log same person within 30s
```

---

## ──────────────────────────────────────────────────
## STEP 3: RUN THE APPLICATION
## ──────────────────────────────────────────────────

```bash
# Start the admin dashboard
streamlit run app.py

# The browser will open automatically at: http://localhost:8501
```

To test camera only (without the dashboard):
```bash
python camera.py
```

---

## ──────────────────────────────────────────────────
## HOW TO USE — STEP BY STEP
## ──────────────────────────────────────────────────

### 1. Login
- Open http://localhost:8501
- Username: `admin`
- Password: `admin123`

### 2. Register a Person
- Go to **Register Person** → **Upload Photo**
- Enter: Person ID (e.g., EMP001), Full Name, Department
- Upload a clear, front-facing photo
- Click **Register Person**

✅ Tips for good registration photos:
- Front-facing, eyes open
- Good lighting (no shadows on face)
- Only one person in the photo
- No sunglasses, heavy filters, or face coverings

### 3. Start Live Recognition
- Go to **Live Camera**
- Select your camera source
- Click **Start Camera**
- The system will recognize registered persons in real time
  - 🟢 Green box = known person (name + confidence shown)
  - 🔴 Red box = unknown person

### 4. View Logs
- Go to **Recognition Logs**
- Filter by date or status
- Export to CSV or Excel

### 5. Manage Persons
- Go to **Manage Persons**
- Edit name/department or delete any person
- Changes take effect immediately

---

## ──────────────────────────────────────────────────
## IP CAMERA / CCTV SETUP
## ──────────────────────────────────────────────────

In config.py or via the Live Camera page, enter:

```
rtsp://username:password@192.168.1.100:554/stream
```

Common RTSP URL formats:
- Hikvision: `rtsp://admin:password@IP:554/Streaming/Channels/101`
- Dahua:     `rtsp://admin:password@IP:554/cam/realmonitor?channel=1&subtype=0`
- Generic:   `rtsp://admin:password@IP:554/stream1`

---

## ──────────────────────────────────────────────────
## PROJECT FILE STRUCTURE
## ──────────────────────────────────────────────────

```
face_recognition_system/
│
├── app.py          ← Admin dashboard (run this)
├── camera.py       ← Live recognition engine
├── database.py     ← All database operations
├── face_utils.py   ← Face encoding & comparison
├── config.py       ← All settings
├── requirements.txt
├── SETUP.md        ← This file
│
├── data/
│   ├── faces.db         ← SQLite database (auto-created)
│   └── face_images/     ← Saved face photos
│       └── snapshots/   ← Unknown person snapshots
│
└── logs/
    └── exports/         ← CSV/Excel exports
```

---

## ──────────────────────────────────────────────────
## TROUBLESHOOTING
## ──────────────────────────────────────────────────

| Problem | Solution |
|---------|----------|
| Camera won't open | Check index (0, 1, 2) in config.py |
| Face not recognized | Improve lighting, re-register with better photo |
| Too many false positives | Lower RECOGNITION_TOLERANCE (e.g., 0.45) |
| Recognition too slow | Check FRAME_SKIP setting (increase to 3 or 4) |
| dlib install fails | Install Visual C++ Build Tools (Windows) or cmake (Linux) |
| "No face detected" on registration | Use a clearer, well-lit front-facing photo |

---

## ──────────────────────────────────────────────────
## FUTURE FEATURES (Planned)
## ──────────────────────────────────────────────────

- [ ] Multiple camera support
- [ ] Anti-spoofing / liveness detection
- [ ] Email/WhatsApp alerts for unknown persons
- [ ] Web-based remote dashboard
- [ ] Door lock / relay integration
- [ ] Cloud database (PostgreSQL)
- [ ] HRMS / school system integration

---

*Built with Python, OpenCV, face_recognition, SQLite, and Streamlit.*
