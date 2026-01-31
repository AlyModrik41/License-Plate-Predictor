import streamlit as st
import tempfile
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="License Plate Detection", layout="centered")
st.title("🚗 License Plate Detection")

# Load model once
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="AlyModrik41/Licence-Plate-Detection",
        filename="bestie.pt"
    )
    return YOLO(model_path)

model = load_model()

# Session state to prevent re-running detection
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:
    # Save uploaded video once
    if "video_path" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            st.session_state.video_path = tmp.name

    st.video(st.session_state.video_path)

    # Run detection ONCE
    if not st.session_state.processed:
        with st.spinner("Detecting license plates..."):
            model.predict(
                source=st.session_state.video_path,
                conf=0.25,
                save=True,
                stream=False
            )
        st.session_state.processed = True
        st.success("Detection completed!")

    st.info("Processed video saved in runs/detect/")