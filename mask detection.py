"""
Mask Detection Web App — Streamlit Cloud Ready
===============================================
No Tk, NO desktop dependencies.
Works fully in browser via Streamlit.

Upload to GitHub → deploy on share.streamlit.io
"""

# ── No Tk imports anywhere ───────────────────────────────────
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io, time, os
from datetime import datetime
from PIL import Image
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Mask Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #0a0e1a; color: #e8eaf0; }

[data-testid="stSidebar"] {
    background: #10152a !important;
    border-right: 1px solid #1f2744;
}

.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #e94560 0%, #f5a623 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1;
}
.hero-sub {
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    color: #555e80; letter-spacing: 0.12em; text-transform: uppercase; margin-top:4px;
}

.status-big {
    font-size: 1.35rem; font-weight: 700; padding: 14px 24px;
    border-radius: 12px; text-align: center; margin: 8px 0;
    font-family: 'Space Mono', monospace;
}
.status-mask     { background:#0d3320; color:#27ae60; border:1px solid #27ae60; }
.status-nomask   { background:#330d0d; color:#e74c3c; border:1px solid #e74c3c; }
.status-improper { background:#332a00; color:#f39c12; border:1px solid #f39c12; }
.status-idle     { background:#13192e; color:#555e80; border:1px solid #1f2744; }

.log-row {
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    padding: 7px 12px; border-radius: 7px; margin-bottom: 4px;
    display: flex; justify-content: space-between; align-items: center;
}
.log-mask     { background:#0d3320; color:#27ae60; }
.log-nomask   { background:#330d0d; color:#e74c3c; }
.log-improper { background:#332a00; color:#f39c12; }

.live-dot {
    display:inline-block; width:10px; height:10px; border-radius:50%;
    background:#e74c3c; animation: blink 1.2s infinite; margin-right:6px;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

.idle-box {
    background:#10152a; border:2px dashed #1f2744; border-radius:14px;
    height:360px; display:flex; align-items:center; justify-content:center;
    flex-direction:column; gap:14px;
}
.idle-icon { font-size:3rem; }
.idle-text {
    font-family:'Space Mono',monospace; color:#555e80;
    font-size:0.82rem; letter-spacing:0.08em;
}

.stButton > button {
    background: #e94560 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    width: 100%;
}
.stButton > button:hover { background: #c73650 !important; }

hr { border-color: #1f2744 !important; }

[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important; color: #e8eaf0 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# FACE TRACKER
# ═══════════════════════════════════════════════════════════════════

class FaceTracker:
    MAX_DISAPPEARED = 30
    MAX_DISTANCE    = 120

    def __init__(self):
        self.next_id = 0
        self.tracks  = {}

    def _cen(self, x1, y1, x2, y2):
        return np.array([(x1+x2)//2, (y1+y2)//2], dtype=float)

    def update(self, detections):
        for tid in self.tracks:
            self.tracks[tid]["gone"] += 1

        stale = [t for t,v in self.tracks.items()
                 if v["gone"] > self.MAX_DISAPPEARED]
        for t in stale:
            del self.tracks[t]

        if not detections:
            return []

        cents  = [self._cen(*d[:4]) for d in detections]
        output = []

        if not self.tracks:
            for i, det in enumerate(detections):
                tid = self._reg(cents[i], det)
                output.append((*det, tid, True))
            return output

        tids   = list(self.tracks.keys())
        tcents = np.array([self.tracks[t]["cen"] for t in tids])
        diffs  = tcents[:,None,:] - np.array(cents)[None,:,:]
        dists  = np.linalg.norm(diffs, axis=2)

        used_t, used_d = set(), set()
        for _ in range(min(len(tids), len(detections))):
            if dists.size == 0: break
            ti, di = np.unravel_index(dists.argmin(), dists.shape)
            if dists[ti, di] > self.MAX_DISTANCE: break
            tid = tids[ti]; used_t.add(ti); used_d.add(di)
            det = detections[di]
            self.tracks[tid].update({"cen": cents[di],
                                     "bbox": det[:4], "gone": 0})
            prev   = self.tracks[tid]["last"]
            is_new = prev != det[4]
            if is_new:
                self.tracks[tid]["last"] = det[4]
            output.append((*det, tid, is_new))
            dists[ti, :] = 1e9
            dists[:, di] = 1e9

        for di, det in enumerate(detections):
            if di not in used_d:
                tid = self._reg(cents[di], det)
                output.append((*det, tid, True))

        return output

    def _reg(self, cen, det):
        tid = self.next_id; self.next_id += 1
        self.tracks[tid] = {"cen": cen, "bbox": det[:4],
                            "gone": 0, "last": det[4]}
        return tid


# ═══════════════════════════════════════════════════════════════════
# SKIN HEURISTIC
# ═══════════════════════════════════════════════════════════════════

_LO = np.array([0,  20,  70], np.uint8)
_HI = np.array([20, 255, 255], np.uint8)

def _skin(roi):
    if roi is None or roi.size == 0: return 0.0
    m = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), _LO, _HI)
    return cv2.countNonZero(m) / max(1, roi.shape[0]*roi.shape[1])

def classify(frame, x1, y1, x2, y2):
    fh = y2-y1; ym = y1+fh//2
    if _skin(frame[ym:y2, x1:x2]) > 0.55:
        return "No Mask", 0.82
    if _skin(frame[y1+int(fh*.45):y1+int(fh*.65), x1:x2]) > 0.40:
        return "Improper Mask", 0.74
    return "With Mask", 0.86


# ═══════════════════════════════════════════════════════════════════
# DETECTOR  (cached so model loads only once)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading detection model…")
def load_detector():
    # Try custom mask model first
    if YOLO_AVAILABLE and os.path.exists("mask_yolov8.pt"):
        try:
            return ("yolo_mask", YOLO("mask_yolov8.pt"))
        except Exception:
            pass
    # Try face model
    if YOLO_AVAILABLE and os.path.exists("yolov8n-face.pt"):
        try:
            return ("yolo_face", YOLO("yolov8n-face.pt"))
        except Exception:
            pass
    # Haar fallback — always works, zero download
    cc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return ("haar", cc)

_LMAP = {
    "with_mask":"With Mask", "mask_weared_correct":"With Mask",
    "mask":"With Mask", "wearing_mask":"With Mask",
    "without_mask":"No Mask", "no_mask":"No Mask",
    "mask_weared_incorrect":"Improper Mask",
    "improper_mask":"Improper Mask", "incorrect_mask":"Improper Mask",
}

def detect(frame, info, conf=0.45):
    kind, model = info
    if kind == "yolo_mask":
        out = []
        for r in model(frame, verbose=False, conf=conf):
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                c  = float(b.conf[0])
                lbl= model.names.get(int(b.cls[0]),"").lower().replace(" ","_")
                out.append((x1,y1,x2,y2, _LMAP.get(lbl,lbl), c))
        return out
    elif kind == "yolo_face":
        out = []
        for r in model(frame, verbose=False, conf=conf):
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                out.append((x1,y1,x2,y2)+classify(frame,x1,y1,x2,y2))
        return out
    else:  # haar
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = model.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
        return [(fx,fy,fx+fw,fy+fh)+classify(frame,fx,fy,fx+fw,fy+fh)
                for (fx,fy,fw,fh) in faces]


# ═══════════════════════════════════════════════════════════════════
# DRAWING
# ═══════════════════════════════════════════════════════════════════

BGRS  = {"With Mask":(0,210,0),"No Mask":(0,0,220),"Improper Mask":(0,165,255)}
EMOJI = {"With Mask":"✅","No Mask":"❌","Improper Mask":"⚠️"}

def draw(frame, tracked):
    for (x1,y1,x2,y2,status,conf,fid,is_new) in tracked:
        bgr = BGRS.get(status,(150,150,150))
        cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,3)
        lbl = f" Face-{fid} | {status}  {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
        cv2.rectangle(frame,(x1,y1-th-16),(x1+tw+4,y1),bgr,-1)
        cv2.putText(frame,lbl,(x1+2,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
        if is_new:
            cv2.putText(frame,"● LOGGED",(x1,y2+22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,bgr,2,cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════

def build_excel(log):
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "Detection Log"
    headers = ["#","Timestamp","Date","Time",
               "Face ID","Status","Confidence (%)","Notes"]
    hfill = PatternFill("solid",fgColor="1F4E79")
    hfont = Font(bold=True,color="FFFFFF",name="Arial",size=11)
    for c,h in enumerate(headers,1):
        cell=ws.cell(row=1,column=c,value=h)
        cell.fill,cell.font=hfill,hfont
        cell.alignment=Alignment(horizontal="center",vertical="center")
    ws.row_dimensions[1].height=25
    for i,w in enumerate([5,22,12,10,9,18,16,22],1):
        ws.column_dimensions[get_column_letter(i)].width=w

    fills={"With Mask":("C6EFCE","276221"),
           "No Mask":("FFC7CE","9C0006"),
           "Improper Mask":("FFEB9C","9C5700")}
    for idx,e in enumerate(log,1):
        bg,fg=fills.get(e["status"],("FFFFFF","000000"))
        fill=PatternFill("solid",fgColor=bg)
        font=Font(color=fg,name="Arial")
        vals=[idx, e["ts"], e["ts"][:10], e["ts"][11:19],
              f"Face-{e['fid']}", e["status"],
              round(e["conf"]*100,1), e.get("notes","")]
        for c,v in enumerate(vals,1):
            cell=ws.cell(row=idx+1,column=c,value=v)
            cell.fill,cell.font=fill,font
            cell.alignment=Alignment(horizontal="center")
    buf=io.BytesIO(); wb.save(buf); buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

def _init():
    for k,v in [("log",[]),
                ("tracker",FaceTracker()),
                ("running",False),
                ("stats",{"With Mask":0,"No Mask":0,"Improper Mask":0}),
                ("uploaded_frame",None)]:
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding:1.2rem 0 0.8rem 0'>
  <div class='hero-title'>🎭 Mask Detection System</div>
  <div class='hero-sub'>Real-time · Per-person tracking · Excel export</div>
</div><hr>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_val = st.slider("Confidence threshold", 0.10, 0.95, 0.45, 0.05)
    mode     = st.radio("Detection mode",
                        ["📷 Live Webcam", "🖼️ Upload Image"],
                        index=0)
    st.markdown("---")

    st.markdown("### 📊 Session Stats")
    c1,c2,c3 = st.columns(3)
    c1.metric("✅ Mask",     st.session_state.stats["With Mask"])
    c2.metric("❌ No Mask",  st.session_state.stats["No Mask"])
    c3.metric("⚠️ Improper", st.session_state.stats["Improper Mask"])
    st.markdown("---")

    st.markdown("### ℹ️ How it works")
    st.info(
        "**Smart per-person tracking:**\n\n"
        "- Each face gets a unique **Face ID**\n"
        "- Logged **once** on first detection\n"
        "- Logged again **only if** status changes\n"
        "  *(e.g. removes or puts on mask)*\n\n"
        "Click **Reset Tracker** to start fresh."
    )
    st.markdown("---")

    if st.button("🔄 Reset Tracker"):
        st.session_state.tracker = FaceTracker()
        st.success("Tracker reset!")

    st.markdown("---")
    if st.session_state.log:
        st.download_button(
            "📥 Download Excel Log",
            data=build_excel(st.session_state.log),
            file_name=f"mask_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.caption("📥 Excel download appears after first detection.")


# ═══════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════

left, right = st.columns([3,2], gap="large")

with right:
    st.markdown("### 📋 Detection Log")
    log_ph = st.empty()

    st.markdown("### 👤 Active Faces")
    face_ph = st.empty()

    def render_log():
        if not st.session_state.log:
            log_ph.info("No detections yet.")
            return
        html = ""
        for e in reversed(st.session_state.log[-18:]):
            css = {"With Mask":"log-mask",
                   "No Mask":"log-nomask",
                   "Improper Mask":"log-improper"}.get(e["status"],"")
            em  = EMOJI.get(e["status"],"")
            html += (f"<div class='log-row {css}'>"
                     f"<span>Face-{e['fid']}&nbsp;&nbsp;"
                     f"{em} {e['status']}</span>"
                     f"<span>{e['ts'][11:19]}&nbsp;{e['conf']:.0%}</span>"
                     f"</div>")
        log_ph.markdown(html, unsafe_allow_html=True)

    render_log()

with left:
    # ── MODE: UPLOAD IMAGE ──────────────────────────────────────────
    if "Upload" in mode:
        st.markdown("### 🖼️ Upload Image for Detection")
        uploaded = st.file_uploader(
            "Upload a photo (JPG/PNG)", type=["jpg","jpeg","png"])

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            detector_info = load_detector()
            raw     = detect(frame, detector_info, conf_val)
            tracked = st.session_state.tracker.update(raw)
            annotated = draw(frame.copy(), tracked)

            # Log new detections
            for (_,_,_,_,status,conf,fid,is_new) in tracked:
                if is_new:
                    st.session_state.stats[status] = \
                        st.session_state.stats.get(status,0)+1
                    st.session_state.log.append({
                        "fid":status,"fid":fid,"status":status,
                        "conf":conf,
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "notes":"Image upload"})

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True)

            if tracked:
                best = max(tracked, key=lambda x:x[5])
                css  = {"With Mask":"status-mask",
                        "No Mask":"status-nomask",
                        "Improper Mask":"status-improper"}.get(best[4],"")
                em   = EMOJI.get(best[4],"")
                st.markdown(
                    f"<div class='status-big {css}'>"
                    f"{em}  {best[4].upper()}  ({best[5]:.0%})</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='status-big status-idle'>"
                    "⬤  No face detected</div>",
                    unsafe_allow_html=True)

            face_ph.info(
                f"👤 {len(st.session_state.tracker.tracks)} face(s) tracked")
            render_log()

    # ── MODE: LIVE WEBCAM ───────────────────────────────────────────
    else:
        st.markdown("### 📷 Live Webcam Detection")

        frame_ph  = st.empty()
        status_ph = st.empty()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Start Detection", use_container_width=True):
                st.session_state.running = True
                st.session_state.tracker = FaceTracker()
        with c2:
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state.running = False

        if not st.session_state.running:
            frame_ph.markdown("""
            <div class='idle-box'>
              <div class='idle-icon'>📷</div>
              <div class='idle-text'>CAMERA FEED WILL APPEAR HERE</div>
              <div style='font-size:0.8rem;color:#3a4470;'>
                Click ▶ Start Detection to begin
              </div>
            </div>""", unsafe_allow_html=True)

        if st.session_state.running:
            detector_info = load_detector()
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error(
                    "❌ **Camera not available.**\n\n"
                    "Streamlit Cloud servers have no webcam.\n\n"
                    "**Use '🖼️ Upload Image' mode instead**, "
                    "or run the app locally with ngrok for live webcam.")
                st.session_state.running = False
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)

                st.markdown(
                    "<div style='margin-bottom:6px'>"
                    "<span class='live-dot'></span>"
                    "<span style='font-family:Space Mono,monospace;"
                    "font-size:0.7rem;color:#e74c3c;letter-spacing:.1em'>"
                    "LIVE</span></div>",
                    unsafe_allow_html=True)

                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret: break

                    try:
                        raw = detect(frame, detector_info, conf_val)
                    except Exception:
                        raw = []

                    tracked   = st.session_state.tracker.update(raw)
                    annotated = draw(frame.copy(), tracked)

                    for (_,_,_,_,status,conf,fid,is_new) in tracked:
                        if is_new:
                            st.session_state.stats[status] = \
                                st.session_state.stats.get(status,0)+1
                            st.session_state.log.append({
                                "fid":fid,"status":status,"conf":conf,
                                "ts":datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"),
                                "notes":"Live webcam"})

                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_ph.image(rgb, channels="RGB",
                                   use_container_width=True)

                    if tracked:
                        best = max(tracked, key=lambda x:x[5])
                        css  = {"With Mask":"status-mask",
                                "No Mask":"status-nomask",
                                "Improper Mask":"status-improper"
                                }.get(best[4],"")
                        em   = EMOJI.get(best[4],"")
                        status_ph.markdown(
                            f"<div class='status-big {css}'>"
                            f"{em}  {best[4].upper()}  "
                            f"({best[5]:.0%})</div>",
                            unsafe_allow_html=True)
                    else:
                        status_ph.markdown(
                            "<div class='status-big status-idle'>"
                            "⬤  No face detected</div>",
                            unsafe_allow_html=True)

                    n = len(st.session_state.tracker.tracks)
                    face_ph.info(f"👤 {n} face(s) currently tracked")
                    render_log()
                    time.sleep(0.04)

                cap.release()