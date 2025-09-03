import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import gdown

# -------------------------------
# Ensure models directory exists
# -------------------------------
os.makedirs("models", exist_ok=True)

# Google Drive file ID for best.pt
BEST_ID = "1KAiUGSIXvpYtw7fWsNzUfuFsCELRGcc2"
BEST_PATH = "models/best.pt"

# Download best.pt if not already present
if not os.path.exists(BEST_PATH):
    url = f"https://drive.google.com/uc?id={BEST_ID}"
    gdown.download(url, BEST_PATH, quiet=False)

# -------------------------------
# Load models
# -------------------------------
road_model = YOLO(BEST_PATH)            # your trained model (damages, speed breakers)
vehicle_model = YOLO("models/yolov8s.pt")  # pretrained COCO model for people/vehicles

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="RoadGuard", layout="wide")
st.title("🚦 RoadGuard — Live & Upload Detection")
st.caption("Detecting: speed breakers, potholes/cracks, humans & vehicles")

# Sidebar
st.sidebar.header("⚙️ Settings")
CONF_DEFAULT = 0.5
conf_thresh = st.sidebar.slider("Confidence", 0.1, 1.0, CONF_DEFAULT, 0.05)
show_road = st.sidebar.checkbox("Detect Road Damages", value=True)
show_veh  = st.sidebar.checkbox("Detect Humans & Vehicles", value=True)

# Drawing colors
ROAD_CIRCLE_COLOR = (0, 0, 255)   # red
SPEED_RECT_COLOR  = (0, 255, 255) # yellow
VEH_RECT_COLOR    = (0, 255, 0)   # green

ROAD_LABELS_SPEED = {
    "speedbreaker", "speed breaker", "speed-bump", "speed_bump",
    "speed hump", "speedbreaker1"
}
VEH_ALLOWED = {"person","car","bus","truck","motorbike","bicycle"}


def draw_road(box, label, frame):
    """Draw speed breaker (rectangle) or pothole/crack (circle)"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    lab = label.lower()
    if any(k in lab for k in ROAD_LABELS_SPEED):
        cv2.rectangle(frame, (x1, y1), (x2, y2), SPEED_RECT_COLOR, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, SPEED_RECT_COLOR, 2)
    else:
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        radius = max((x2 - x1)//2, (y2 - y1)//2)
        cv2.circle(frame, (cx, cy), max(8, radius), ROAD_CIRCLE_COLOR, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ROAD_CIRCLE_COLOR, 2)


def run_detection_both(frame_bgr, conf):
    """Apply both models"""
    out = frame_bgr.copy()

    if show_road:
        rr = road_model.predict(out, conf=conf, device="cpu", verbose=False)
        for r in rr:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                label = road_model.names.get(cls_id, "road")
                draw_road(b, label, out)

    if show_veh:
        vr = vehicle_model.predict(out, conf=conf, device="cpu", verbose=False)
        for r in vr:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                label = vehicle_model.names.get(cls_id, "obj")
                if label in VEH_ALLOWED:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cv2.rectangle(out, (x1, y1), (x2, y2), VEH_RECT_COLOR, 2)
                    cv2.putText(out, label, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, VEH_RECT_COLOR, 2)
    return out


# -------------------------------
# MODE SELECTOR
# -------------------------------
mode = st.radio("Choose Mode", ["📷 Live Camera (WebRTC)", "🖼 Upload Image", "🎞 Upload Video"], horizontal=True)

# -------------------------------
# 1) LIVE CAMERA (WebRTC)
# -------------------------------
if mode == "📷 Live Camera (WebRTC)":
    st.info("Allow camera permission in your browser. If video does not appear, try another browser/network.")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    def _callback(frame):
        img = frame.to_ndarray(format="bgr24")
        out = run_detection_both(img, conf_thresh)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    webrtc_streamer(
        key="roadguard-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=_callback,
        async_processing=True,
    )

# -------------------------------
# 2) IMAGE UPLOAD
# -------------------------------
elif mode == "🖼 Upload Image":
    up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if up:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            tf.write(up.read())
            tmp_path = tf.name

        img = cv2.imread(tmp_path)
        if img is not None:
            res = run_detection_both(img, conf_thresh)
            st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_column_width=True)

            out_path = "annotated_image.jpg"
            cv2.imwrite(out_path, res)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download Result", f, file_name="annotated_image.jpg", mime="image/jpeg")

        os.remove(tmp_path)

# -------------------------------
# 3) VIDEO UPLOAD
# -------------------------------
else:
    upv = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
    if upv:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
            tf.write(upv.read())
            vpath = tf.name

        cap = cv2.VideoCapture(vpath)
        if cap.isOpened():
            stframe = st.empty()
            pbar = st.progress(0)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            out_path = "annotated_video.mp4"
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                res = run_detection_both(frame, conf_thresh)
                out.write(res)
                stframe.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_column_width=True)

                i += 1
                if total > 0:
                    pbar.progress(min(100, int(i * 100 / total)))

            cap.release()
            out.release()
            pbar.empty()

            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download Annotated Video", f, file_name="annotated_video.mp4", mime="video/mp4")

        os.remove(vpath)
