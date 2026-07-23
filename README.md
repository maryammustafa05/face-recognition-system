# 🎯 Smart Face Recognition System

A real-time face recognition system that converts a normal webcam or IP camera into a smart monitoring camera. Built with Python, DeepFace (ArcFace), OpenCV, and Streamlit.

---

## ✨ Features

- 🎥 Live camera feed with real-time face detection
- 👤 Face registration via photo upload or live camera capture
- ✅ Automatic recognition of registered persons
- ❓ Unknown person detection with snapshot saving
- 🔔 Pending approvals — admin reviews and registers unknown faces
- 📋 Recognition logs with date, time, and camera location
- 📤 Export logs to CSV and Excel
- 🔒 Secure admin login with bcrypt hashed passwords
- 🔄 Camera auto-reconnect on disconnection
- ⚙️ All settings configurable from one file

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| DeepFace + ArcFace | Face recognition (512-number embeddings) |
| OpenCV | Camera feed and video processing |
| SQLite | Database for persons, logs, and approvals |
| Streamlit | Web-based admin dashboard |
| bcrypt | Secure password hashing |
| Pandas + OpenPyXL | Log export to CSV and Excel |

---

## 📁 Project Structure

```
face_recognition_system/
│
├── app.py              ← Admin dashboard (run this)
├── camera.py           ← Live recognition engine
├── database.py         ← All database operations
├── face_utils.py       ← Face encoding and comparison
├── config.py           ← All settings
├── requirements.txt    ← Python dependencies
├── SETUP.md            ← Setup and installation guide
│
├── data/
│   ├── faces.db            ← SQLite database (auto-created)
│   └── face_images/        ← Saved face photos and snapshots
│
└── logs/
    └── exports/            ← CSV and Excel exports
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Open in Browser
```
http://localhost:8501
```

### 4. Login
```
Username: admin
Password: admin123
```

> ⚠️ Change the default password in `config.py` before deploying.

---

## 📸 How It Works

1. **Register** a person with their photo and details
2. System converts their face into a unique **512-number embedding** using ArcFace
3. **Live camera** detects faces in every frame
4. Each detected face is compared against all registered embeddings
5. **Known person** → name and confidence shown on screen, attendance logged
6. **Unknown person** → snapshot saved, added to Pending Approvals
7. Admin **reviews** unknown faces and registers or rejects them

---

## 🗄️ Database Structure

| Table | Purpose |
|---|---|
| `persons` | Registered persons with face embeddings |
| `recognition_logs` | All recognition events with timestamps |
| `admins` | Admin users with hashed passwords |
| `pending_approvals` | Unknown faces waiting for admin review |

---

## ⚙️ Configuration

All settings are in `config.py`:

```python
CAMERA_SOURCE = 0                        # 0 = webcam, 1 = USB, "rtsp://..." = IP camera
RECOGNITION_TOLERANCE = 0.63            # Recognition threshold (lower = stricter)
DUPLICATE_LOG_INTERVAL_SECONDS = 60     # Prevent duplicate logs within 60 seconds
ADMIN_USERNAME = "admin"                 # Change before deploying
ADMIN_PASSWORD = "admin123"              # Change before deploying
```

---

## 📋 Requirements

- Python 3.8 – 3.12
- Webcam or IP camera
- 4GB RAM minimum (8GB recommended)
- Windows 10+ / Ubuntu 20.04+ / macOS 12+

---

## 🚀 Future Enhancements

- [ ] Multiple camera support
- [ ] Email/WhatsApp alerts for unknown persons
- [ ] PostgreSQL for enterprise scale
- [ ] Anti-spoofing / liveness detection
- [ ] Door lock integration
- [ ] Mobile app
- [ ] HRMS/ERP integration

---

## 📄 License

This project is developed for internal company use.
