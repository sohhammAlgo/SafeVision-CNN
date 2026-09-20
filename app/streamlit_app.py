import json
import cv2
import numpy as np
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

from src.detection.pipeline import SafetyPipeline
from src.detection.visualization import draw_detections
from src.db.database import initialize_database
from src.db.repository import (
    get_recent_violations,
    get_violation_statistics
)
from src.explainability.gradcam import GradCAM


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
# Load Safety Pipeline
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
# Load Grad-CAM
# ============================================

@st.cache_resource
def load_gradcam():
    return GradCAM()


try:

    gradcam = load_gradcam()

except Exception as error:

    st.error(
        f"Grad-CAM initialization failed: {error}"
    )

    st.stop()


# ============================================
# Live Camera Processor
# ============================================

class SafeVisionProcessor(VideoProcessorBase):

    def __init__(self):

        self.pipeline = pipeline
        self.confidence = confidence

    def recv(self, frame):

        # WebRTC frame → OpenCV BGR
        image = frame.to_ndarray(
            format="bgr24"
        )

        # Process frame
        pipeline_output = (
            self.pipeline.process_frame(
                image,
                confidence=self.confidence
            )
        )

        # Draw detections
        annotated_frame = draw_detections(
            image,
            pipeline_output
        )

        # Return processed frame
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


# ============================================
# Violation Summary
# ============================================

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


# ============================================
# PPE Violation Breakdown
# ============================================

st.subheader("📊 PPE Violation Breakdown")


summary, ppe_rows = (
    get_violation_statistics()
)


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


ppe_col1, ppe_col2, ppe_col3 = (
    st.columns(3)
)


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

st.subheader("📋 Violation History")


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
            f"**Missing:** "
            f"{', '.join(missing_ppe)} | "
            f"**Severity:** {row['severity']} | "
            f"**Time:** {row['timestamp']}"
        )


# ============================================
# AI Explainability
# ============================================

st.divider()

st.subheader("🧠 AI Explainability")


st.write(
    "Upload a face image to visualize "
    "which regions influenced the Mask CNN prediction."
)


uploaded_image = st.file_uploader(
    "Upload a mask/face image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_image is not None:

    file_bytes = np.asarray(
        bytearray(
            uploaded_image.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Unable to read the uploaded image."
        )

    else:

        try:

            overlay, prediction, confidence_score, heatmap = (
                gradcam.generate(image)
            )

            explain_col1, explain_col2 = (
                st.columns(2)
            )

            with explain_col1:

                st.image(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Original Image",
                    use_container_width=True
                )

            with explain_col2:

                st.image(
                    cv2.cvtColor(
                        overlay,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Grad-CAM Explanation",
                    use_container_width=True
                )

            st.success(
                f"Prediction: **{prediction}**  \n"
                f"Confidence: **{confidence_score:.2%}**"
            )

        except Exception as error:

            st.error(
                f"Grad-CAM failed: {error}"
            )