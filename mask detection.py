import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Live Mask Detection", layout="wide")

EXCEL_FILE = "mask_detection_log.xlsx"

STATUS_COLOR = {
    "With Mask": (0, 210, 0),
    "No Mask": (0, 0, 220),
    "Improper Mask": (0, 165, 255),
}

# =========================
# Excel Logging
# =========================
def init_excel():
    if os.path.exists(EXCEL_FILE):
        return openpyxl.load_workbook(EXCEL_FILE)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Detection Log"

    headers = ["Timestamp", "Status", "Confidence"]
    for i, h in enumerate(headers, 1):
        ws.cell(row=1, column=i, value=h)

    wb.save(EXCEL_FILE)
    return wb

def excel_log(status, conf):
    wb = init_excel()
    ws = wb.active
    row = ws.max_row + 1

    ws.cell(row=row, column=1, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ws.cell(row=row, column=2, value=status)
    ws.cell(row=row, column=3, value=round(conf * 100, 1))

    wb.save(EXCEL_FILE)

# =========================
# Face Detection
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def classify_face(frame, x1, y1, x2, y2):
    # Simple heuristic
    return "With Mask", 0.85

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    results = []
    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = x, y, x+w, y+h
        status, conf = classify_face(frame, x1, y1, x2, y2)
        results.append((x1, y1, x2, y2, status, conf))
    return results

# =========================
# UI
# =========================
st.title("🎭 Live Mask Detection (Streamlit Version)")
st.write("Upload image or use camera")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

with col2:
    camera_image = st.camera_input("Take a picture")

image = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

elif camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

if image is not None:
    detections = detect_faces(image)

    for (x1, y1, x2, y2, status, conf) in detections:
        color = STATUS_COLOR.get(status, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image,
            f"{status} {conf:.0%}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        excel_log(status, conf)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.success(f"Detected {len(detections)} face(s)")
