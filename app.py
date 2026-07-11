import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, date
from PIL import Image

from config import (
    APP_TITLE, APP_ICON, ADMIN_USERNAME, ADMIN_PASSWORD,
    RECOGNITION_TOLERANCE, FACE_IMAGES_DIR
)
from database import (
    initialize_database, seed_admin,
    register_person, get_all_persons, get_person_by_id,
    update_person, soft_delete_person,
    get_all_logs, get_log_summary,
    verify_admin,
    get_pending_approvals, get_pending_count,
    approve_pending, reject_pending
)
from face_utils import (
    get_face_encoding_from_image, get_face_encoding_from_frame,
    save_face_image, validate_image_has_face
)
from camera import FaceRecognitionCamera

# ─── App Setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize DB on first run
initialize_database()
seed_admin(ADMIN_USERNAME, ADMIN_PASSWORD)


# ─── Session State Initialization ─────────────────────────────────────────────

def init_session():
    defaults = {
        "logged_in": False,
        "camera": None,
        "camera_running": False,
        "page": "Dashboard",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════

def login_page():
    """Render the login form."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("---")
        st.markdown(f"# {APP_ICON} {APP_TITLE}")
        st.markdown("### Admin Login")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter admin username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("🔓 Login", use_container_width=True)

            if submitted:
                if verify_admin(username, password):
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

        st.markdown("---")
        st.info("Default credentials: admin / admin123  \n*(Change in config.py before deploying)*")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.markdown("---")

        pages = {
            "📊 Dashboard":        "Dashboard",
            "📷 Live Camera":      "Live Camera",
            "👤 Register Person":  "Register",
            "👥 Manage Persons":   "Manage",
            "📋 Recognition Logs": "Logs",
            "⚙️ Settings":         "Settings",
        }

        # Add pending count badge
        pending = get_pending_count()
        if pending > 0:
            pages[f"🔔 Pending Approvals ({pending})"] = "Pending"
        else:
            pages["🔔 Pending Approvals"] = "Pending"

        for label, page in pages.items():
            if st.button(label, use_container_width=True,
                         type="primary" if st.session_state.page == page else "secondary"):
                st.session_state.page = page
                st.rerun()

        st.markdown("---")

        # Camera status indicator
        if st.session_state.camera_running:
            st.success("🟢 Camera: Running")
            cam = st.session_state.camera
            if cam:
                st.caption(f"FPS: {cam.fps}")
        else:
            st.warning("🔴 Camera: Stopped")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            stop_camera()
            st.session_state.logged_in = False
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def start_camera(source=None):
    """Start the recognition camera."""
    if st.session_state.camera_running:
        return
    cam = FaceRecognitionCamera(source=source)
    cam.start()
    st.session_state.camera = cam
    st.session_state.camera_running = True


def stop_camera():
    """Stop the recognition camera."""
    if st.session_state.camera and st.session_state.camera_running:
        st.session_state.camera.stop()
    st.session_state.camera = None
    st.session_state.camera_running = False


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.title("📊 System Dashboard")

    summary = get_log_summary()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Registered Persons", summary["registered_persons"])
    col2.metric("📋 Total Recognitions", summary["total_logs"])
    col3.metric("✅ Known Recognized",   summary["known"])
    col4.metric("⚠️ Unknown Detected",   summary["unknown"])

    st.markdown("---")

    # Recent activity
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🕐 Recent Recognition Activity")
        logs = get_all_logs(limit=10)
        if logs:
            log_data = [dict(row) for row in logs]
            df = pd.DataFrame(log_data)[
                ["person_name", "person_id", "department", "status", "confidence", "recognized_at"]
            ]
            df.columns = ["Name", "ID", "Dept", "Status", "Confidence %", "Time"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No recognition events yet. Start the camera to begin.")

    with col_right:
        st.subheader("👤 Registered Persons")
        persons = get_all_persons()
        if persons:
            for p in persons[:8]:
                st.markdown(f"**{p['name']}** `{p['person_id']}`  \n_{p['department']}_")
                st.markdown("---")
        else:
            st.info("No persons registered yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE CAMERA
# ═══════════════════════════════════════════════════════════════════════════════

def page_live_camera():
    st.title("📷 Live Face Recognition")

    col_ctrl, col_info = st.columns([1, 2])

    with col_ctrl:
        source_option = st.selectbox(
            "Camera Source",
            ["Webcam (0)", "USB Camera (1)", "IP Camera (RTSP)"]
        )

        source = 0
        if source_option == "USB Camera (1)":
            source = 1
        elif source_option == "IP Camera (RTSP)":
            source = st.text_input("RTSP URL", placeholder="rtsp://user:pass@192.168.1.1:554/stream")

        if not st.session_state.camera_running:
            if st.button("▶️ Start Camera", use_container_width=True, type="primary"):
                start_camera(source=source)
                st.rerun()
        else:
            if st.button("⏹️ Stop Camera", use_container_width=True):
                stop_camera()
                st.rerun()

            if st.button("🔄 Reload Face Data", use_container_width=True):
                if st.session_state.camera:
                    st.session_state.camera.reload_faces()
                    st.success("Face data reloaded!")

    # Status panel (placeholder so it updates in the loop)
    with col_info:
        status_box = st.empty()

    st.markdown("---")

    if not st.session_state.camera_running or not st.session_state.camera:
        st.info("👆 Press **Start Camera** to begin live recognition.")
        st.markdown("""
        **Camera Tips:**
        - Ensure good front-facing lighting
        - Stay 0.5–2m from the camera
        - Avoid backlighting (window behind you)
        - The system processes every other frame for better performance
        """)
        return

    # ── Live streaming loop ───────────────────────────────────────────────────
    # LEARNING NOTE:
    #   Streamlit's st.empty() creates a placeholder that can be overwritten.
    #   By updating it in a while loop we get a true live video feed —
    #   each iteration replaces the previous image in the same spot on screen.

    frame_placeholder = st.empty()
    error_placeholder = st.empty()

    while st.session_state.camera_running:
        cam = st.session_state.camera
        if cam is None:
            break

        # Show error if camera lost
        if cam.last_error:
            error_placeholder.error(f"⚠️ {cam.last_error}")
        else:
            error_placeholder.empty()

        # Get latest frame from background thread
        frame = cam.get_current_frame()

        if frame is not None:
            # OpenCV uses BGR — Streamlit needs RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the video placeholder (this is what makes it "live")
            frame_placeholder.image(
                rgb_frame,
                channels="RGB",
                use_column_width=True,
                caption=f"Live Feed  •  FPS: {cam.fps}"
            )

            # Update recognition status panel
            results = cam.get_current_results()
            if results:
                status_lines = []
                for r in results:
                    if r["status"] == "Known":
                        status_lines.append(f"✅ **{r['name']}** — {r['confidence']}% confidence")
                    else:
                        status_lines.append("❓ **Unknown Person** detected")
                status_box.success("\n\n".join(status_lines))
            else:
                status_box.info("Camera running. Waiting for faces...")

        # ~20 FPS display rate — fast enough to look smooth, not too heavy
        time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REGISTER PERSON
# ═══════════════════════════════════════════════════════════════════════════════

def page_register():
    st.title("👤 Register New Person")

    tab1, tab2 = st.tabs(["📁 Upload Photo", "📷 Capture from Camera"])

    # ── Tab 1: Upload Image ───────────────────────────────────────────────────
    with tab1:
        st.markdown("Upload a clear, front-facing photo with good lighting.")

        col1, col2 = st.columns([1, 1])

        with col1:
            person_id   = st.text_input("Person ID *", placeholder="e.g., EMP001 or STU2024")
            name        = st.text_input("Full Name *", placeholder="e.g., Ahmed Ali")
            department  = st.text_input("Department / Category", placeholder="e.g., Engineering")
            uploaded    = st.file_uploader("Face Photo *", type=["jpg", "jpeg", "png"])

        with col2:
            if uploaded:
                st.image(uploaded, caption="Uploaded photo", width=250)

        if st.button("✅ Register Person", type="primary", disabled=not uploaded):
            if not person_id or not name:
                st.error("Person ID and Name are required.")
            elif not person_id.strip().isalnum():
                st.error("Person ID must be alphanumeric (letters and numbers only).")
            else:
                with st.spinner("Processing face..."):
                    # Save the uploaded file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded.getbuffer())
                        tmp_path = tmp.name

                    # Extract face encoding
                    encoding, msg = get_face_encoding_from_image(tmp_path)

                    if encoding is None:
                        st.error(f"❌ {msg}")
                        os.unlink(tmp_path)
                    else:
                        # Save image permanently
                        image_path = save_face_image(tmp_path, person_id.strip(), is_file=True)
                        os.unlink(tmp_path)

                        # Register in DB
                        success, message = register_person(
                            person_id=person_id.strip(),
                            name=name.strip(),
                            department=department.strip(),
                            image_path=image_path,
                            face_encoding=encoding
                        )

                        if success:
                            st.success(message)
                            # Reload camera face data if running
                            if st.session_state.camera_running and st.session_state.camera:
                                st.session_state.camera.reload_faces()
                        else:
                            st.error(message)

    # ── Tab 2: Live Capture ───────────────────────────────────────────────────
    with tab2:
        st.markdown("Use your webcam to capture the face directly.")

        if not st.session_state.camera_running:
            st.warning("⚠️ Start the Live Camera first (from the **Live Camera** page).")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                cap_id   = st.text_input("Person ID *", key="cap_id")
                cap_name = st.text_input("Full Name *", key="cap_name")
                cap_dept = st.text_input("Department", key="cap_dept")

            if st.button("📸 Capture & Register", type="primary"):
                if not cap_id or not cap_name:
                    st.error("Person ID and Name are required.")
                else:
                    cam = st.session_state.camera
                    frame = cam.get_current_frame() if cam else None

                    if frame is None:
                        st.error("Could not get frame from camera.")
                    else:
                        encoding, msg = get_face_encoding_from_frame(frame)

                        if encoding is None:
                            st.error(f"❌ {msg}")
                        else:
                            image_path = save_face_image(frame, cap_id.strip(), is_file=False)
                            success, message = register_person(
                                person_id=cap_id.strip(),
                                name=cap_name.strip(),
                                department=cap_dept.strip(),
                                image_path=image_path,
                                face_encoding=encoding
                            )

                            if success:
                                st.success(message)
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.image(rgb, caption="Captured photo", width=250)
                                cam.reload_faces()
                            else:
                                st.error(message)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MANAGE PERSONS
# ═══════════════════════════════════════════════════════════════════════════════

def page_manage():
    st.title("👥 Manage Registered Persons")

    persons = get_all_persons()

    if not persons:
        st.info("No persons registered yet. Go to **Register Person** to add someone.")
        return

    st.write(f"**{len(persons)} registered person(s)**")

    for person in persons:
        with st.expander(f"👤 {person['name']}  |  ID: {person['person_id']}  |  {person['department']}"):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                # Show face image if available
                if person["image_path"] and os.path.exists(person["image_path"]):
                    st.image(person["image_path"], width=150)
                else:
                    st.caption("No image available")

            with col2:
                st.markdown(f"**Name:** {person['name']}")
                st.markdown(f"**ID:** `{person['person_id']}`")
                st.markdown(f"**Department:** {person['department']}")
                st.markdown(f"**Registered:** {person['registered_at']}")

            with col3:
                edit_key = f"edit_{person['person_id']}"
                if st.button("✏️ Edit", key=f"btn_{edit_key}"):
                    st.session_state[edit_key] = True

                if st.button("🗑️ Delete", key=f"del_{person['person_id']}"):
                    if soft_delete_person(person["person_id"]):
                        st.success(f"'{person['name']}' removed.")
                        if st.session_state.camera_running and st.session_state.camera:
                            st.session_state.camera.reload_faces()
                        st.rerun()

            # Edit form
            if st.session_state.get(f"edit_{person['person_id']}", False):
                with st.form(f"form_{person['person_id']}"):
                    new_name = st.text_input("Name", value=person["name"])
                    new_dept = st.text_input("Department", value=person["department"])
                    if st.form_submit_button("💾 Save Changes"):
                        if update_person(person["person_id"], new_name, new_dept):
                            st.success("Updated successfully!")
                            st.session_state[f"edit_{person['person_id']}"] = False
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOGNITION LOGS
# ═══════════════════════════════════════════════════════════════════════════════

def page_logs():
    st.title("📋 Recognition Logs")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        date_filter = st.date_input("Filter by Date", value=None)
    with col2:
        status_filter = st.selectbox("Filter by Status", ["All", "Known", "Unknown"])
    with col3:
        limit = st.selectbox("Max Rows", [100, 250, 500, 1000], index=1)

    date_str = date_filter.strftime("%Y-%m-%d") if date_filter else None
    logs = get_all_logs(date_filter=date_str, status_filter=status_filter, limit=limit)

    st.write(f"**{len(logs)} log entries found**")

    if logs:
        df = pd.DataFrame([dict(row) for row in logs])

        # Style status column
        def style_status(val):
            color = "#28a745" if val == "Known" else "#dc3545"
            return f"color: {color}; font-weight: bold"

        display_cols = ["person_name", "person_id", "department",
                        "status", "confidence", "camera_name", "recognized_at"]
        display_df = df[display_cols].copy()
        display_df.columns = ["Name", "ID", "Dept", "Status", "Conf %", "Camera", "Time"]

        st.dataframe(
            display_df.style.applymap(style_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True
        )

        # Export options
        st.markdown("---")
        st.subheader("📤 Export Logs")

        col_csv, col_excel = st.columns(2)

        with col_csv:
            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name=f"recognition_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_excel:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                display_df.to_excel(writer, sheet_name="Recognition Logs", index=False)
            st.download_button(
                "⬇️ Download Excel",
                data=buffer.getvalue(),
                file_name=f"recognition_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.info("No logs found with the selected filters.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

def page_settings():
    st.title("⚙️ System Settings")

    st.info("ℹ️ To permanently change settings, edit **config.py** and restart the app.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recognition Settings")
        st.markdown(f"""
        | Setting | Current Value |
        |---------|--------------|
        | Tolerance | `{RECOGNITION_TOLERANCE}` (lower = stricter) |
        | Detection Model | `hog` (CPU-friendly) |
        | Frame Skip | Every 2nd frame |
        | Min Face Size | 60 px |
        | Duplicate Interval | 30 seconds |
        """)

    with col2:
        st.subheader("Quick Diagnostics")

        if st.button("🧪 Test Camera"):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                st.success("✅ Webcam accessible (index 0)")
                cap.release()
            else:
                st.error("❌ Cannot access webcam index 0")

        if st.button("🗄️ Database Info"):
            summary = get_log_summary()
            st.json(summary)

        if st.button("📂 Open Face Images Folder"):
            st.code(FACE_IMAGES_DIR)

    st.markdown("---")
    st.subheader("📖 How Recognition Works")
    st.markdown("""
    1. **Frame captured** from camera every ~33ms
    2. **Face detected** using HOG (Histogram of Oriented Gradients) model
    3. **128-point encoding** generated — a mathematical "fingerprint" of the face
    4. **Compared** against all stored encodings using Euclidean distance
    5. If distance ≤ tolerance (`{:.2f}`), the person is **recognized**
    6. Result is **annotated** on screen with name and confidence
    7. **Log saved** to database (duplicate prevention: 30s window)
    """.format(RECOGNITION_TOLERANCE))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PENDING APPROVALS
# ═══════════════════════════════════════════════════════════════════════════════

def page_pending_approvals():
    st.title("🔔 Pending Approvals")
    st.markdown("Unknown faces detected by the camera. Review each one and either register them or reject.")

    pending = get_pending_approvals()

    if not pending:
        st.success("✅ No pending approvals. All unknown faces have been reviewed.")
        return

    st.write(f"**{len(pending)} unknown face(s) waiting for review**")
    st.markdown("---")

    for record in pending:
        col_img, col_form = st.columns([1, 2])

        with col_img:
            # Show the captured snapshot
            if record["snapshot_path"] and os.path.exists(record["snapshot_path"]):
                st.image(record["snapshot_path"], caption=f"Detected at {record['detected_at']}", width=200)
            else:
                st.warning("Snapshot not available")
            st.caption(f"📷 Camera: {record['camera_name']}")
            st.caption(f"🕐 Time: {record['detected_at']}")

        with col_form:
            st.markdown("**Register this person or reject:**")

            with st.form(key=f"approval_form_{record['id']}"):
                person_id  = st.text_input("Person ID *",   placeholder="e.g. EMP005")
                name       = st.text_input("Full Name *",   placeholder="e.g. Ahmed Khan")
                department = st.text_input("Department",    placeholder="e.g. IT")

                col_approve, col_reject = st.columns(2)

                with col_approve:
                    approve = st.form_submit_button("✅ Approve & Register", use_container_width=True, type="primary")

                with col_reject:
                    reject = st.form_submit_button("❌ Reject", use_container_width=True)

                if approve:
                    if not person_id or not name:
                        st.error("Person ID and Name are required.")
                    else:
                        success, message = approve_pending(
                            pending_id=record["id"],
                            person_id=person_id.strip(),
                            name=name.strip(),
                            department=department.strip()
                        )
                        if success:
                            st.success(f"✅ {name} registered successfully!")
                            # Reload camera face data
                            if st.session_state.camera_running and st.session_state.camera:
                                st.session_state.camera.reload_faces()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(message)

                if reject:
                    reject_pending(record["id"])
                    st.info("Rejected and removed from pending list.")
                    time.sleep(0.5)
                    st.rerun()

        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not st.session_state.logged_in:
        login_page()
        return

    render_sidebar()

    page = st.session_state.page

    if page == "Dashboard":
        page_dashboard()
    elif page == "Live Camera":
        page_live_camera()
    elif page == "Register":
        page_register()
    elif page == "Manage":
        page_manage()
    elif page == "Logs":
        page_logs()
    elif page == "Settings":
        page_settings()
    elif page == "Pending":
        page_pending_approvals()


if __name__ == "__main__":
    main()