import json
import cv2
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from src.detection.pipeline import SafetyPipeline
from src.detection.visualization import draw_detections
from src.db.database import initialize_database
from src.db.repository import (
    get_recent_violations,
    get_violation_statistics
)


# ============================================
# Page configuration
# ============================================

st.set_page_config(
    page_title="SafeVision",
    page_icon="🦺",
    layout="wide"
)


# ============================================
# Initialize database
# ============================================

initialize_database()


# ============================================
# Header
# ============================================

st.title("🦺 SafeVision")

st.write(
    "Real-Time PPE & Face Safety Detection System"
)

st.divider()


# ============================================
# Sidebar
# ============================================

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)


# ============================================
# Load Pipeline
# ============================================

@st.cache_resource
def load_pipeline():
    return SafetyPipeline()


try:
    pipeline = load_pipeline()

except FileNotFoundError as error:

    st.error(
        f"Model not found: {error}"
    )

    st.stop()


# ============================================
# Live Camera Processor
# ============================================

class SafeVisionProcessor(VideoProcessorBase):

    def __init__(self):
        self.pipeline = pipeline

    def recv(self, frame):

        # WebRTC frame → OpenCV BGR
        image = frame.to_ndarray(format="bgr24")

        # Process through SafeVision pipeline
        pipeline_output = self.pipeline.process_frame(
            image,
            confidence=confidence
        )

        # Draw detections
        annotated_frame = draw_detections(
            image,
            pipeline_output
        )

        # OpenCV BGR → WebRTC frame
        return frame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# ============================================
# Live Camera
# ============================================

st.subheader("📷 Live Camera")

webrtc_streamer(
    key="safevision-camera",
    video_processor_factory=SafeVisionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


st.divider()


# ============================================
# Violation Dashboard
# ============================================

st.subheader("⚠️ Recent Violations")

if st.button("🔄 Refresh Dashboard"):
    st.rerun()
violations = get_recent_violations(
    limit=20
)


col1, col2, col3 = st.columns(3)


total = len(violations)

high = sum(
    1
    for row in violations
    if row["severity"] == "HIGH"
)

critical = sum(
    1
    for row in violations
    if row["severity"] == "CRITICAL"
)


with col1:

    st.metric(
        "Total Violations",
        total
    )


with col2:

    st.metric(
        "High",
        high
    )


with col3:

    st.metric(
        "Critical",
        critical
    )

st.subheader("📊 PPE Violation Breakdown")

summary, ppe_rows = get_violation_statistics()

helmet_violations = 0
mask_violations = 0
vest_violations = 0

for row in ppe_rows:

    missing_ppe = json.loads(
        row["missing_ppe"]
    )

    count = row["count"]

    if "helmet" in missing_ppe:
        helmet_violations += count

    if "mask" in missing_ppe:
        mask_violations += count

    if "vest" in missing_ppe:
        vest_violations += count


ppe_col1, ppe_col2, ppe_col3 = st.columns(3)

with ppe_col1:
    st.metric(
        "🪖 Helmet Violations",
        helmet_violations
    )

with ppe_col2:
    st.metric(
        "😷 Mask Violations",
        mask_violations
    )

with ppe_col3:
    st.metric(
        "🦺 Vest Violations",
        vest_violations
    )


# ============================================
# Violation List
# ============================================

if not violations:

    st.info(
        "No violations recorded."
    )

else:

    for row in violations:

        missing_ppe = json.loads(
            row["missing_ppe"]
        )

        st.write(
            f"**Track ID:** {row['track_id']} | "
            f"**Missing:** {', '.join(missing_ppe)} | "
            f"**Severity:** {row['severity']} | "
            f"**Time:** {row['timestamp']}"
        )