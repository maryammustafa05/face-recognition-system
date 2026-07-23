# Smart Face Recognition System — User Manual

---

## Logging In

1. Open your browser and go to `http://localhost:8501`
2. Enter your username and password
3. Click **Login**

Default credentials:
- Username: `admin`
- Password: `admin123`

---

## Dashboard

After login you will see the Dashboard. It shows:

- Total registered persons
- Total recognition events
- How many were Known
- How many were Unknown
- Recent recognition activity table

This gives you a quick overview of everything happening in the system.

---

## Registering a New Person

1. Click **Register Person** in the sidebar
2. You have two options:

**Option A — Upload a Photo**
- Enter the Person ID (example: EMP001)
- Enter the Full Name
- Enter the Department
- Click **Browse files** and select a clear front facing photo
- Click **Register Person**

**Option B — Capture from Live Camera**
- Make sure the camera is running first (go to Live Camera and start it)
- Come back to Register Person and click the **Capture from Camera** tab
- Enter Person ID, Full Name, and Department
- Click **Capture and Register**

Tips for a good registration photo:
- Face should be clearly visible and front facing
- Good lighting with no shadows on the face
- Only one person in the photo
- No sunglasses or face coverings

---

## Live Camera

1. Click **Live Camera** in the sidebar
2. Select your camera source from the dropdown
3. Click **Start Camera**

The live feed will appear showing:
- **Green box with name** — recognized person
- **Red box** — unknown person
- Confidence percentage shown next to the name

To stop the camera click **Stop Camera**.

If you register a new person while the camera is running click **Reload Face Data** so the camera recognizes the new person immediately.

---

## Pending Approvals

When an unknown person is detected by the camera their face snapshot is automatically saved and appears here for your review.

If there are pending approvals the sidebar will show a number badge like **Pending Approvals (3)**.

To review:
1. Click **Pending Approvals** in the sidebar
2. You will see the face snapshot with the detection time and camera name
3. If you know this person fill in their Person ID, Full Name, and Department
4. Click **Approve and Register** — they will be immediately added to the system and recognized going forward
5. If you do not want to register them click **Reject** to remove them from the list

---

## Managing Registered Persons

1. Click **Manage Persons** in the sidebar
2. You will see all registered persons with their photos and details
3. Click on a person to expand their details

To edit a person:
- Click **Edit**
- Update the name or department
- Click **Save Changes**

To remove a person:
- Click **Delete**
- They will be marked inactive and will no longer be recognized
- Their historical logs are kept for records

---

## Recognition Logs

1. Click **Recognition Logs** in the sidebar
2. You will see all recognition events

You can filter by:
- **Date** — select a specific date to see that day's activity
- **Status** — filter by Known or Unknown only

**Exporting Logs:**
- Click **Download CSV** to export as a CSV file
- Click **Download Excel** to export as an Excel file

Both exports include name, ID, department, status, confidence, camera, and timestamp.

---

## Settings

Click **Settings** in the sidebar to view current system configuration.

You can also run quick diagnostics:
- **Test Camera** — checks if your webcam is accessible
- **Database Info** — shows current count of logs and persons

To change any setting open `config.py` in the project folder and update the value, then restart the app.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Camera not starting | Check if another app is using the camera (Zoom, Teams etc) and close it |
| Face not recognized | Improve lighting or register the person again with a better photo |
| Too many unknown detections | Lower the threshold in config.py from 0.63 to 0.58 |
| App not opening in browser | Make sure you ran `streamlit run app.py` in the terminal |
| Pending approvals not showing | Make sure camera is running and unknown faces are being detected |

---

*Smart Face Recognition System — Admin User Manual*
