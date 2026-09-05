"""
Face & Emotion Analytics Dashboard
-----------------------------------
Live face detection + emotion recognition (DeepFace) with a real-time
analytics dashboard built in Streamlit.

Run with:  streamlit run app.py
"""

import time
import numpy as np
import cv2
import av
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from deepface import DeepFace
from utils import (
    log_emotion, load_log, clear_log, session_summary,
    draw_face_box, EMOTION_COLORS,
)

# ----------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------
st.set_page_config(page_title="Emotion Analytics", page_icon="🎭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 700 !important; }

.stApp {
    background: radial-gradient(circle at 20% 0%, #241a2e 0%, #17111d 60%, #120d17 100%);
    color: #EDE7F0;
}

section[data-testid="stSidebar"] {
    background-color: #1B1420;
    border-right: 1px solid #33263e;
}

div[data-testid="stMetric"] {
    background: #221a2c;
    border: 1px solid #3a2c48;
    border-left: 4px solid #F4A300;
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="stMetricLabel"] { color: #b9aec4; }

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
}

hr { border-color: #33263e; }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Face & Emotion Analytics Dashboard")
st.caption("Real-time face detection + emotion recognition, with session analytics.")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    analyze_every_n = st.slider("Analyze every Nth frame", 2, 15, 5,
                                 help="Higher = faster/smoother video, lower = more responsive emotion updates.")
    st.divider()
    if st.button("🗑️ Clear session log", use_container_width=True):
        clear_log()
        st.success("Log cleared.")
    st.divider()
    st.caption("Built with Streamlit · OpenCV · DeepFace")

# ----------------------------------------------------------------------
# Live webcam processor
# ----------------------------------------------------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.skip = analyze_every_n
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.cached_labels = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        do_analyze = (self.frame_count % self.skip == 0) and len(faces) > 0
        labels = []

        for i, (x, y, w, h) in enumerate(faces):
            label = self.cached_labels[i] if i < len(self.cached_labels) else "detecting…"
            if do_analyze:
                try:
                    crop = img[y:y + h, x:x + w]
                    res = DeepFace.analyze(
                        crop, actions=["emotion"], enforce_detection=False,
                        detector_backend="skip", silent=True,
                    )
                    res = res[0] if isinstance(res, list) else res
                    label = res["dominant_emotion"]
                    log_emotion(label, res["emotion"], len(faces))
                except Exception:
                    pass
            labels.append(label)
            color = _emotion_bgr(label)
            draw_face_box(img, x, y, w, h, label, color=color)

        if do_analyze:
            self.cached_labels = labels

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def _emotion_bgr(label):
    hex_color = EMOTION_COLORS.get(label, "#FFBD59").lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)  # OpenCV uses BGR


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_live, tab_upload, tab_analytics = st.tabs(
    ["📹 Live Webcam", "🖼️ Analyze an Image", "📊 Analytics"]
)

with tab_live:
    st.write("Click **Start** below and allow camera access. Detected faces are boxed "
              "and labeled with the dominant emotion in real time.")
    webrtc_streamer(
        key="emotion-live",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.info("Every detection is logged automatically — check the **Analytics** tab for trends.")

with tab_upload:
    st.write("Upload a photo to run a full facial analysis (emotion, age, gender).")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing face(s)…"):
            try:
                results = DeepFace.analyze(
                    img, actions=["emotion", "age", "gender"],
                    enforce_detection=False, silent=True,
                )
                if not isinstance(results, list):
                    results = [results]

                annotated = img.copy()
                for res in results:
                    x, y, w, h = (res["region"]["x"], res["region"]["y"],
                                  res["region"]["w"], res["region"]["h"])
                    label = f"{res['dominant_emotion']} · {int(res['age'])}y · {res['dominant_gender']}"
                    color = _emotion_bgr(res["dominant_emotion"])
                    draw_face_box(annotated, x, y, w, h, label, color=color)
                    log_emotion(res["dominant_emotion"], res["emotion"], len(results), force=True)

                col1, col2 = st.columns([1.3, 1])
                with col1:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                              caption="Detected faces", use_container_width=True)
                with col2:
                    for res in results:
                        st.subheader(f"😀 {res['dominant_emotion'].title()}")
                        scores = res["emotion"]
                        fig = go.Figure(go.Bar(
                            x=list(scores.values()), y=list(scores.keys()),
                            orientation="h",
                            marker_color=[EMOTION_COLORS.get(k, "#F4A300") for k in scores.keys()],
                        ))
                        fig.update_layout(
                            height=280, margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#EDE7F0", xaxis_title="Confidence (%)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Estimated age: {int(res['age'])} · Gender: {res['dominant_gender']}")
            except Exception as e:
                st.error(f"Couldn't analyze this image: {e}")

with tab_analytics:
    df = load_log()
    st.subheader("Session Overview")

    if df.empty:
        st.warning("No detections logged yet. Use the Live Webcam or Analyze an Image tab first.")
    else:
        summary = session_summary(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total detections", summary["total"])
        c2.metric("Most common emotion", summary["top_emotion"].title())
        c3.metric("Session length", f"{summary['duration_min']} min")
        c4.metric("Avg. faces / frame", summary["avg_faces"])

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Emotion distribution**")
            counts = df["dominant_emotion"].value_counts().reset_index()
            counts.columns = ["emotion", "count"]
            fig_pie = px.pie(
                counts, names="emotion", values="count", hole=0.55,
                color="emotion", color_discrete_map=EMOTION_COLORS,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#EDE7F0",
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("**Emotion intensity over time**")
            emotion_cols = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            fig_line = go.Figure()
            for col in emotion_cols:
                if col in df.columns:
                    fig_line.add_trace(go.Scatter(
                        x=df["timestamp"], y=df[col], mode="lines", name=col,
                        line=dict(color=EMOTION_COLORS.get(col, "#fff"), width=2),
                        stackgroup=None,
                    ))
            fig_line.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#EDE7F0", legend=dict(orientation="h", y=-0.25),
                yaxis_title="Confidence (%)", height=380,
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.divider()
        st.markdown("**Recent detections**")
        st.dataframe(df.tail(25).sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button(
            "⬇️ Download full log (CSV)",
            data=df.to_csv(index=False),
            file_name="emotion_log.csv",
            mime="text/csv",
        )
