import sqlite3
import json
import os
from datetime import datetime
from config import DB_PATH


def get_connection():
    """
    Create and return a database connection.
    'check_same_thread=False' allows use across multiple threads (needed for Streamlit).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Rows behave like dicts: row["name"] instead of row[0]
    return conn


def initialize_database():
    """
    Create all tables if they don't exist yet.
    Called once when the app starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── Table 1: Registered Persons ──────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   TEXT UNIQUE NOT NULL,      -- Employee/Student ID
            name        TEXT NOT NULL,
            department  TEXT DEFAULT '',
            image_path  TEXT DEFAULT '',
            face_encoding TEXT NOT NULL,           -- JSON list of 128 numbers
            registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active   INTEGER DEFAULT 1          -- 1=active, 0=deleted
        )
    """)

    # ── Table 2: Recognition Logs ─────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id     TEXT DEFAULT 'UNKNOWN',
            person_name   TEXT DEFAULT 'Unknown Person',
            department    TEXT DEFAULT '',
            recognized_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_name   TEXT DEFAULT 'Main Camera',
            status        TEXT DEFAULT 'Unknown',  -- 'Known' or 'Unknown'
            confidence    REAL DEFAULT 0.0,        -- Match confidence %
            snapshot_path TEXT DEFAULT ''          -- Optional captured image
        )
    """)

    # ── Table 3: Admin Users ──────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # ── Table 4: Pending Approvals ────────────────────────────────────────────
    # Stores unknown faces detected by camera waiting for admin review
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_approvals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_path TEXT NOT NULL,         -- Photo of unknown face
            face_encoding TEXT NOT NULL,         -- 512-number embedding
            detected_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_name   TEXT DEFAULT 'Main Camera',
            status        TEXT DEFAULT 'pending' -- 'pending', 'approved', 'rejected'
        )
    """)

    cursor.execute("""
         CREATE TABLE IF NOT EXISTS face_angles(
                   id            INTEGER PRIMARY KEY AUTOINCREMENT,
                   person_id     TEXT NOT NULL,
                   face_encoding TEXT NOT NULL,
                   angle_label   TEXT DEFAULT 'front', --front,left,right etc
                   image_path    TEXT DEFAULT '',
                   added_at      DATETIME DEFAULT CURRENT_TIMESTAMP
                   )
    """)
    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


# ═══════════════════════════════════════════════════════════════════════════════
# PERSON MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def register_person(person_id, name, department, image_path, face_encoding):
    """
    Save a new person to the database.

    face_encoding: list of 128 floats — converted to JSON string for storage.
    Returns: (success: bool, message: str)
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        encoding_json = json.dumps(face_encoding.tolist())  # numpy → list → JSON
        cursor.execute("""
            INSERT INTO persons (person_id, name, department, image_path, face_encoding)
            VALUES (?, ?, ?, ?, ?)
        """, (person_id, name, department, image_path, encoding_json))
        cursor.execute("""
        INSERT INTO face_angles (person_id, face_encoding, angle_label, image_path)
            VALUES (?, ?, 'front', ?)
        """, (person_id, encoding_json, image_path))
        conn.commit()
        return True, f"✅ '{name}' registered successfully!"
    except sqlite3.IntegrityError:
        return False, f"⚠️ Person ID '{person_id}' already exists."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"
    finally:
        conn.close()


def get_all_persons(include_inactive=False):
    """
    Retrieve all registered persons.
    Returns a list of sqlite3.Row objects (accessible like dicts).
    """
    conn = get_connection()
    cursor = conn.cursor()
    if include_inactive:
        cursor.execute("SELECT * FROM persons ORDER BY registered_at DESC")
    else:
        cursor.execute("SELECT * FROM persons WHERE is_active=1 ORDER BY registered_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_person_by_id(person_id):
    """Get a single person by their person_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM persons WHERE person_id=? AND is_active=1", (person_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def update_person(person_id, name, department):
    """Update a person's name and department."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE persons SET name=?, department=? WHERE person_id=?
    """, (name, department, person_id))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def soft_delete_person(person_id):
    """
    Mark person as inactive instead of permanently deleting.
    This preserves historical log data.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE persons SET is_active=0 WHERE person_id=?", (person_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def load_all_face_encodings():
    """
    Load all active persons' face encodings into memory.

    Returns:
        encodings: list of numpy arrays (128-float vectors)
        metadata:  list of dicts with name/id/department

    LEARNING NOTE:
        Face recognition works by comparing 128-number "fingerprints" of faces.
        We load all of them into RAM so comparisons happen instantly.
    """
    import numpy as np
    persons = get_all_persons()
    encodings = []
    metadata = []

    for person in persons:
        try:
            encoding = np.array(json.loads(person["face_encoding"]))
            encodings.append(encoding)
            metadata.append({
                "person_id":  person["person_id"],
                "name":       person["name"],
                "department": person["department"],
                "image_path": person["image_path"],
            })
        except Exception as e:
            print(f"[DB] Warning: Could not load encoding for {person['name']}: {e}")

    return encodings, metadata


# ═══════════════════════════════════════════════════════════════════════════════
# LOG MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def add_recognition_log(person_id, person_name, department, camera_name,
                         status, confidence=0.0, snapshot_path=""):
    """Save a recognition event to the log table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO recognition_logs
            (person_id, person_name, department, camera_name, status, confidence, snapshot_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (person_id, person_name, department, camera_name, status, confidence, snapshot_path))
    conn.commit()
    conn.close()


def get_recent_log_time(person_id):
    """
    Get the most recent log timestamp for a person.
    Used to prevent duplicate log entries.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT recognized_at FROM recognition_logs
        WHERE person_id=?
        ORDER BY recognized_at DESC LIMIT 1
    """, (person_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return datetime.strptime(row["recognized_at"], "%Y-%m-%d %H:%M:%S")
    return None


def get_all_logs(date_filter=None, status_filter=None, limit=500):
    """
    Retrieve recognition logs with optional filters.

    date_filter: "YYYY-MM-DD" string
    status_filter: "Known" or "Unknown"
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM recognition_logs WHERE 1=1"
    params = []

    if date_filter:
        query += " AND DATE(recognized_at) = ?"
        params.append(date_filter)

    if status_filter and status_filter != "All":
        query += " AND status = ?"
        params.append(status_filter)

    query += " ORDER BY recognized_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_log_summary():
    """Return count statistics for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM recognition_logs")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as known FROM recognition_logs WHERE status='Known'")
    known = cursor.fetchone()["known"]

    cursor.execute("SELECT COUNT(*) as unknown FROM recognition_logs WHERE status='Unknown'")
    unknown = cursor.fetchone()["unknown"]

    cursor.execute("SELECT COUNT(*) as persons FROM persons WHERE is_active=1")
    persons = cursor.fetchone()["persons"]

    conn.close()
    return {"total_logs": total, "known": known, "unknown": unknown, "registered_persons": persons}


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN AUTH
# ═══════════════════════════════════════════════════════════════════════════════

def verify_admin(username, password):
    """Check admin credentials. Returns True/False."""
    import bcrypt
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM admins WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return False
    stored = row["password"]
    # Support both hashed and plain text passwords during transition
    try:
        return bcrypt.checkpw(password.encode(), stored if isinstance(stored, bytes) else stored.encode())
    except Exception:
        return stored == password


def seed_admin(username, password):
    """Insert the default admin with hashed password if not exists."""
    import bcrypt
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE username=?", (username,))
    existing = cursor.fetchone()
    if not existing:
        cursor.execute(
            "INSERT INTO admins (username, password) VALUES (?, ?)",
            (username, hashed)
        )
        conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PENDING APPROVALS
# ═══════════════════════════════════════════════════════════════════════════════

def add_pending_approval(snapshot_path, face_encoding, camera_name="Main Camera"):
    """
    Save an unknown face for admin review.
    Called automatically when an unknown person is detected.
    """
    import numpy as np
    conn = get_connection()
    cursor = conn.cursor()

    encoding_json = json.dumps(
        face_encoding.tolist() if hasattr(face_encoding, 'tolist') else list(face_encoding)
    )

    cursor.execute("""
        INSERT INTO pending_approvals (snapshot_path, face_encoding, camera_name)
        VALUES (?, ?, ?)
    """, (snapshot_path, encoding_json, camera_name))
    conn.commit()
    conn.close()


def get_pending_approvals():
    """Get all pending (unreviewed) unknown faces."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM pending_approvals
        WHERE status = 'pending'
        ORDER BY detected_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_pending_count():
    """Return count of pending approvals — used for badge in sidebar."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM pending_approvals WHERE status='pending'")
    count = cursor.fetchone()["cnt"]
    conn.close()
    return count


def approve_pending(pending_id, person_id, name, department):
    """
    Admin approves an unknown face — registers them as a new person.
    Returns (success, message).
    """
    import numpy as np
    conn = get_connection()
    cursor = conn.cursor()

    # Get the pending record
    cursor.execute("SELECT * FROM pending_approvals WHERE id=?", (pending_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False, "Pending record not found."

    # Register the person using their stored encoding and snapshot
    encoding = np.array(json.loads(row["face_encoding"]))
    success, message = register_person(
        person_id=person_id,
        name=name,
        department=department,
        image_path=row["snapshot_path"],
        face_encoding=encoding
    )

    if success:
        # Mark as approved
        conn2 = get_connection()
        cursor2 = conn2.cursor()
        cursor2.execute(
            "UPDATE pending_approvals SET status='approved' WHERE id=?",
            (pending_id,)
        )
        conn2.commit()
        conn2.close()

    return success, message


def reject_pending(pending_id):
    """Admin rejects an unknown face — marks it as rejected."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE pending_approvals SET status='rejected' WHERE id=?",
        (pending_id,)
    )
    conn.commit()
    conn.close()

def add_face_angle(person_id,face_encoding,angle_label,image_path=""):
        """Add an additional face angle for an existing person."""
        conn=get_connection()
        cursor=conn.cursor()
        try:
            encoding_json=json.dumps(
                face_encoding.tolist() if hasattr(face_encoding,'tolist') else list(face_encoding)
            )
            cursor.execute("""
               INSERT INTO face_angles (person_id, face_encoding, angle_label, image_path)
               VALUES (?, ?, ?, ?)
            """,(person_id, encoding_json, angle_label, image_path))
            conn.commit()
            return True, f"✅ Angle '{angle_label}' added successfully!"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
        finally:
            conn.close()
    
def get_face_angles(person_id):
        conn=get_connection()
        cursor=conn.cursor()
        cursor.execute(""" 
         SELECT * FROM face_angles WHERE person_id=? ORDER BY added_at ASC
         """,(person_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    
def load_all_face_encodings():
        import numpy as np
        conn=get_connection()
        cursor=conn.cursor()
        cursor.execute("""
        SELECT fa.face_encoding, fa.angle_label, p.person_id, p.name, p.department, p.image_path
        FROM face_angles fa
        JOIN persons p ON fa.person_id = p.person_id
        WHERE p.is_active = 1
        ORDER BY p.person_id, fa.added_at
    """)
        rows=cursor.fetchall()
        conn.close()
        encodings=[]
        metadata=[]
        for row in rows:
         try:
            encoding = np.array(json.loads(row["face_encoding"]))
            encodings.append(encoding)
            metadata.append({
                "person_id":  row["person_id"],
                "name":       row["name"],
                "department": row["department"],
                "image_path": row["image_path"],
                "angle":      row["angle_label"],
            })
         except Exception as e:
            print(f"[DB] Warning: Could not load angle for {row['person_id']}: {e}")

        print(f"[DB] Loaded {len(encodings)} face angle(s) for {len(set(m['person_id'] for m in metadata))} person(s).")
        return encodings, metadata