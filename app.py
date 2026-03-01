"""
Mask Detection Web App
Streamlit Cloud compatible — no tkinter, no ultralytics
"""
import streamlit as st
import cv2
import numpy as np
import io
from datetime import datetime
from PIL import Image
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Mask Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf0; }
[data-testid="stSidebar"] { background: #10152a !important; border-right: 1px solid #1f2744; }
.hero { font-size:2.2rem; font-weight:800;
        background:linear-gradient(135deg,#e94560,#f5a623);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub  { font-family:'Space Mono',monospace; font-size:.72rem;
        color:#555e80; letter-spacing:.12em; text-transform:uppercase; }
.status-box { font-size:1.3rem; font-weight:700; padding:14px 20px;
              border-radius:12px; text-align:center; margin:8px 0;
              font-family:'Space Mono',monospace; }
.s-mask     { background:#0d3320; color:#27ae60; border:1px solid #27ae60; }
.s-nomask   { background:#330d0d; color:#e74c3c; border:1px solid #e74c3c; }
.s-improper { background:#332a00; color:#f39c12; border:1px solid #f39c12; }
.s-none     { background:#13192e; color:#555e80; border:1px solid #1f2744; }
.log-row    { font-family:'Space Mono',monospace; font-size:.75rem;
              padding:7px 12px; border-radius:7px; margin-bottom:4px;
              display:flex; justify-content:space-between; }
.l-mask     { background:#0d3320; color:#27ae60; }
.l-nomask   { background:#330d0d; color:#e74c3c; }
.l-improper { background:#332a00; color:#f39c12; }
hr { border-color:#1f2744 !important; }
.stButton > button { background:#e94560 !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    width:100%; padding:.5rem 1rem !important; }
.stButton > button:hover { background:#c73650 !important; }
[data-testid="stMetricValue"] { font-family:'Space Mono',monospace !important;
    font-size:1.8rem !important; color:#e8eaf0 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# FACE TRACKER
# ═══════════════════════════════════════════════════════════════════
class FaceTracker:
    MAX_GONE = 30
    MAX_DIST = 120

    def __init__(self):
        self.nid    = 0
        self.tracks = {}

    def _cen(self, x1, y1, x2, y2):
        return np.array([(x1+x2)//2, (y1+y2)//2], dtype=float)

    def update(self, dets):
        for t in self.tracks:
            self.tracks[t]["gone"] += 1
        for t in [k for k,v in self.tracks.items() if v["gone"] > self.MAX_GONE]:
            del self.tracks[t]

        if not dets:
            return []

        cents = [self._cen(*d[:4]) for d in dets]
        out   = []

        if not self.tracks:
            for i, d in enumerate(dets):
                tid = self._reg(cents[i], d)
                out.append((*d, tid, True))
            return out

        tids   = list(self.tracks.keys())
        tcents = np.array([self.tracks[t]["cen"] for t in tids])
        dists  = np.linalg.norm(
            tcents[:,None,:] - np.array(cents)[None,:,:], axis=2)

        used_t, used_d = set(), set()
        for _ in range(min(len(tids), len(dets))):
            if dists.size == 0: break
            ti, di = np.unravel_index(dists.argmin(), dists.shape)
            if dists[ti, di] > self.MAX_DIST: break
            tid = tids[ti]; used_t.add(ti); used_d.add(di)
            d   = dets[di]
            self.tracks[tid].update({"cen": cents[di], "gone": 0})
            prev   = self.tracks[tid]["last"]
            is_new = prev != d[4]
            if is_new: self.tracks[tid]["last"] = d[4]
            out.append((*d, tid, is_new))
            dists[ti,:] = 1e9; dists[:,di] = 1e9

        for di, d in enumerate(dets):
            if di not in used_d:
                tid = self._reg(cents[di], d)
                out.append((*d, tid, True))
        return out

    def _reg(self, cen, d):
        tid = self.nid; self.nid += 1
        self.tracks[tid] = {"cen": cen, "gone": 0, "last": d[4]}
        return tid


# ═══════════════════════════════════════════════════════════════════
# SKIN HEURISTIC DETECTOR
# ═══════════════════════════════════════════════════════════════════
_LO = np.array([0,  20,  70], np.uint8)
_HI = np.array([20, 255, 255], np.uint8)

def _skin(roi):
    if roi is None or roi.size == 0: return 0.0
    m = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), _LO, _HI)
    return cv2.countNonZero(m) / max(1, roi.shape[0] * roi.shape[1])

def classify(frame, x1, y1, x2, y2):
    fh = y2 - y1; ym = y1 + fh // 2
    if _skin(frame[ym:y2, x1:x2]) > 0.55:
        return "No Mask", 0.82
    if _skin(frame[y1+int(fh*.45) : y1+int(fh*.65), x1:x2]) > 0.40:
        return "Improper Mask", 0.74
    return "With Mask", 0.86


@st.cache_resource(show_spinner="Loading face detector…")
def get_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def run_detect(frame, conf_thresh):
    det  = get_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    out = []
    for (fx, fy, fw, fh) in faces:
        status, conf = classify(frame, fx, fy, fx+fw, fy+fh)
        if conf >= conf_thresh:
            out.append((fx, fy, fx+fw, fy+fh, status, conf))
    return out


# ═══════════════════════════════════════════════════════════════════
# DRAW
# ═══════════════════════════════════════════════════════════════════
BGRS  = {"With Mask":(0,210,0), "No Mask":(0,0,220), "Improper Mask":(0,165,255)}
EMOJI = {"With Mask":"✅", "No Mask":"❌", "Improper Mask":"⚠️"}
CSS   = {"With Mask":"s-mask", "No Mask":"s-nomask", "Improper Mask":"s-improper"}
LCSS  = {"With Mask":"l-mask", "No Mask":"l-nomask", "Improper Mask":"l-improper"}

def draw(frame, tracked):
    for (x1,y1,x2,y2,status,conf,fid,is_new) in tracked:
        bgr = BGRS.get(status, (150,150,150))
        cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 3)
        lbl = f" Face-{fid} | {status}  {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1,y1-th-16), (x1+tw+4,y1), bgr, -1)
        cv2.putText(frame, lbl, (x1+2,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        if is_new:
            cv2.putText(frame, "LOGGED", (x1, y2+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════
# EXCEL
# ═══════════════════════════════════════════════════════════════════
def build_excel(log):
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "Detection Log"
    headers = ["#","Timestamp","Date","Time","Face ID",
               "Status","Confidence (%)","Notes"]
    hf = PatternFill("solid", fgColor="1F4E79")
    hfont = Font(bold=True, color="FFFFFF", name="Arial", size=11)
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill=hf; cell.font=hfont
        cell.alignment=Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 25
    for i, w in enumerate([5,22,12,10,9,18,16,22], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    fills = {"With Mask":("C6EFCE","276221"),
             "No Mask":("FFC7CE","9C0006"),
             "Improper Mask":("FFEB9C","9C5700")}
    for idx, e in enumerate(log, 1):
        bg, fg = fills.get(e["status"], ("FFFFFF","000000"))
        fill = PatternFill("solid", fgColor=bg)
        font = Font(color=fg, name="Arial")
        vals = [idx, e["ts"], e["ts"][:10], e["ts"][11:19],
                f"Face-{e['fid']}", e["status"],
                round(e["conf"]*100,1), "Auto-detected"]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row=idx+1, column=c, value=v)
            cell.fill=fill; cell.font=font
            cell.alignment=Alignment(horizontal="center")
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
for k, v in [("log",[]), ("tracker",FaceTracker()),
             ("stats",{"With Mask":0,"No Mask":0,"Improper Mask":0})]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:1rem 0 .5rem'>
  <div class='hero'>🎭 Mask Detection System</div>
  <div class='sub'>Real-time · Per-person tracking · Excel export</div>
</div><hr>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_val = st.slider("Confidence threshold", 0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Mask",     st.session_state.stats["With Mask"])
    c2.metric("❌ No Mask",  st.session_state.stats["No Mask"])
    c3.metric("⚠️ Improper", st.session_state.stats["Improper Mask"])

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.info(
        "Each face gets a **Face ID**.\n\n"
        "Logged **once** on first detection.\n"
        "Logged again **only if status changes**\n"
        "*(mask removed or put on)*"
    )
    if st.button("🔄 Reset Tracker"):
        st.session_state.tracker = FaceTracker()
        st.session_state.log     = []
        st.session_state.stats   = {"With Mask":0,"No Mask":0,"Improper Mask":0}
        st.success("Reset done!")

    st.markdown("---")
    if st.session_state.log:
        st.download_button(
            "📥 Download Excel Log",
            data=build_excel(st.session_state.log),
            file_name=f"mask_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    else:
        st.caption("Excel download appears after first detection.")


# ═══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════
left, right = st.columns([3, 2], gap="large")

with right:
    st.markdown("### 📋 Detection Log")
    log_ph = st.empty()

    def render_log():
        if not st.session_state.log:
            log_ph.info("No detections yet. Upload an image to start!"); return
        html = ""
        for e in reversed(st.session_state.log[-20:]):
            lc = LCSS.get(e["status"], "")
            em = EMOJI.get(e["status"], "")
            html += (f"<div class='log-row {lc}'>"
                     f"<span>Face-{e['fid']}&nbsp;&nbsp;{em} {e['status']}</span>"
                     f"<span>{e['ts'][11:19]}&nbsp;{e['conf']:.0%}</span></div>")
        log_ph.markdown(html, unsafe_allow_html=True)

    render_log()

    st.markdown("### 👤 Active Faces")
    faces_ph = st.empty()
    n = len(st.session_state.tracker.tracks)
    faces_ph.info(f"👤 {n} face(s) currently tracked")


with left:
    st.markdown("### 🖼️ Upload Image or Use Camera")

    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Take Photo"])

    # ── TAB 1: UPLOAD ──────────────────────────────────────────────
    with tab1:
        uploaded = st.file_uploader(
            "Upload a photo (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed")

        if uploaded:
            data  = np.frombuffer(uploaded.read(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            raw     = run_detect(frame, conf_val)
            tracked = st.session_state.tracker.update(raw)
            annotated = draw(frame.copy(), tracked)

            # Log new status changes
            for (_,_,_,_,status,conf,fid,is_new) in tracked:
                if is_new:
                    st.session_state.stats[status] = \
                        st.session_state.stats.get(status, 0) + 1
                    st.session_state.log.append({
                        "fid":fid, "status":status, "conf":conf,
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True)

            if tracked:
                best  = max(tracked, key=lambda x: x[5])
                bstat = best[4]
                st.markdown(
                    f"<div class='status-box {CSS.get(bstat,'s-none')}'>"
                    f"{EMOJI.get(bstat,'')}  {bstat.upper()}  "
                    f"({best[5]:.0%})</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='status-box s-none'>⬤  No face detected</div>",
                    unsafe_allow_html=True)

            n = len(st.session_state.tracker.tracks)
            faces_ph.info(f"👤 {n} face(s) tracked")
            render_log()

        else:
            st.markdown("""
            <div style='background:#10152a;border:2px dashed #1f2744;
                        border-radius:14px;height:300px;display:flex;
                        align-items:center;justify-content:center;
                        flex-direction:column;gap:12px;'>
              <div style='font-size:2.5rem'>📁</div>
              <div style='font-family:Space Mono,monospace;color:#555e80;
                          font-size:.8rem;letter-spacing:.08em;'>
                DROP IMAGE HERE OR CLICK TO UPLOAD
              </div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 2: CAMERA ──────────────────────────────────────────────
    with tab2:
        st.info(
            "📸 Take a photo with your device camera.\n\n"
            "The app will instantly detect mask status.")

        cam_img = st.camera_input("Point camera at face and click 📸",
                                  label_visibility="collapsed")

        if cam_img:
            data  = np.frombuffer(cam_img.read(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            raw     = run_detect(frame, conf_val)
            tracked = st.session_state.tracker.update(raw)
            annotated = draw(frame.copy(), tracked)

            for (_,_,_,_,status,conf,fid,is_new) in tracked:
                if is_new:
                    st.session_state.stats[status] = \
                        st.session_state.stats.get(status, 0) + 1
                    st.session_state.log.append({
                        "fid":fid, "status":status, "conf":conf,
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True)

            if tracked:
                best  = max(tracked, key=lambda x: x[5])
                bstat = best[4]
                st.markdown(
                    f"<div class='status-box {CSS.get(bstat,'s-none')}'>"
                    f"{EMOJI.get(bstat,'')}  {bstat.upper()}  "
                    f"({best[5]:.0%})</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='status-box s-none'>⬤  No face detected</div>",
                    unsafe_allow_html=True)

            n = len(st.session_state.tracker.tracks)
            faces_ph.info(f"👤 {n} face(s) tracked")
            render_log()
