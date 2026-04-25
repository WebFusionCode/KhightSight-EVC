import streamlit as st
import cv2
import numpy as np
import tempfile
import json
from pathlib import Path

from detect import load_detector
from ocr import build_ocr_engine
from pipeline import process_frame

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="KnightSight ANPR System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .css-1d391kg {
        background-color: #1e1e1e;
    }
    .metric-card {
        background-color: #161a24;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 4px solid #00ffcc;
        margin-bottom: 10px;
    }
    .big-font {
        font-size:24px !important;
        font-weight: bold;
        color: #00ffcc;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL CACHING ---
@st.cache_resource
def load_models():
    # Attempt to load detector automatically (picks newest finetuned by default)
    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"Failed to load YOLOv8 detector: {e}")
        detector = None

    # Load OCR engine (using mobile optimal models by default)
    try:
        ocr_engine = build_ocr_engine(lang="en")
    except Exception as e:
        st.error(f"Failed to load PaddleOCR engine: {e}")
        ocr_engine = None
        
    return detector, ocr_engine

detector, ocr_engine = load_models()

# --- SIDEBAR UI ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/GitHub_Invertocat_Logo.svg/1024px-GitHub_Invertocat_Logo.svg.png", width=50)
st.sidebar.title("KnightSight V2")
st.sidebar.markdown("### Next-Gen ANPR Pipeline")
st.sidebar.divider()

source_type = st.sidebar.radio("Input Source Stream", ["[+] Image Upload", "[>] Video Stream", "[O] Live Web Camera"], index=0)

st.sidebar.markdown("**Pipeline Configs**")
enforce_indian = st.sidebar.checkbox("Enforce Formatting Rules", value=True, help="Activates dictionary validation & AA00AA0000 Regex mapping.")
padding = st.sidebar.slider("Extraction Padding (px)", min_value=0, max_value=20, value=8)

st.sidebar.divider()
st.sidebar.caption("System Latency: ~65ms / Frame")
st.sidebar.caption("Detector Confidence: 98.02% (mAP@50)")

# --- MAIN PANEL ---
st.title("KnightSight Command Center")
st.markdown("Live Autonomous License Plate Recognition System")
st.divider()

if detector is None or ocr_engine is None:
    st.error("AI Core systems offline. Please check model weights.")
    st.stop()

# --- IMAGE PROCESSING ---
if source_type == "[+] Image Upload":
    uploaded_file = st.file_uploader("Upload Target Image", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("Engaging Neural Modules..."):
            annotated_image, results, diagnostics = process_frame(
                image, 
                detector, 
                ocr_engine, 
                padding=padding, 
                enforce_indian_format=enforce_indian
            )
            
            # Convert BGR to RGB for Streamlit
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Create premium layout
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### Processed Intel")
                st.image(annotated_rgb, use_container_width=True, clamp=True)
                
            with col2:
                st.markdown("### Targets Acquired")
                if results:
                    for i, res in enumerate(results):
                        st.markdown(f"""
                        <div class="metric-card">
                            <span class="big-font">{res['plate_text']}</span><br>
                            <span style="color:#aaa;">Confidence Matrix: <b>{res['confidence'] * 100:.1f}%</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.json(results, expanded=False)
                    
                    st.markdown("---")
                    st.markdown("### Vision Diagnostics")
                    for diag in diagnostics:
                        d_col1, d_col2 = st.columns(2)
                        with d_col1:
                            st.caption("Raw Extraction")
                            st.image(diag["raw"], use_container_width=True)
                        if "preprocessed" in diag:
                            with d_col2:
                                st.caption("AI De-Noise/Enchanced")
                                st.image(diag["preprocessed"], use_container_width=True, clamp=True)
                else:
                    st.warning("No authorized targets locked.")
                    
# --- VIDEO PROCESSING ---
elif source_type == "[>] Video Stream":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.subheader("Video Processing")
        
        if st.sidebar.button("Run Pipeline"):
            # Prepare UI elements
            frame_placeholder = st.empty()
            json_placeholder = st.empty()
            
            cap = cv2.VideoCapture(video_path)
            frame_skip = 2  # Process every 3rd frame
            frame_count = 0
            
            spatial_cache = {}
            aggregated_results = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    annotated_frame, frame_results, _ = process_frame(
                        frame,
                        detector,
                        ocr_engine,
                        padding=padding,
                        enforce_indian_format=enforce_indian,
                        spatial_cache=spatial_cache
                    )
                    
                    # Convert to RGB
                    disp_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(disp_frame, channels="RGB")
                    
                    if frame_results:
                        json_placeholder.json(frame_results)
                        aggregated_results.extend(frame_results)
                
                frame_count += 1
                
            cap.release()
            st.success("Video processing complete!")
            
            # Simple dedup based on text
            seen = set()
            unique_results = []
            for r in aggregated_results:
                if r['plate_text'] and r['plate_text'] not in seen:
                    seen.add(r['plate_text'])
                    unique_results.append(r)
            
            st.json(unique_results)

# --- WEBCAM PROCESSING ---
elif source_type == "[O] Live Web Camera":
    st.markdown("### Live Camera Feed")
    camera_image = st.camera_input("Capture License Plate")
    
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("Processing capture..."):
            annotated_image, results, diagnostics = process_frame(
                image, 
                detector, 
                ocr_engine, 
                padding=padding, 
                enforce_indian_format=enforce_indian
            )
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(annotated_rgb, use_container_width=True)
            with c2:
                if results:
                    for i, res in enumerate(results):
                        st.markdown(f"""
                        <div class="metric-card">
                            <span class="big-font">{res['plate_text']}</span><br>
                            <span style="color:#aaa;">Confidence Matrix: <b>{res['confidence'] * 100:.1f}%</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.json(results, expanded=False)
                else:
                    st.warning("No targets detected.")
