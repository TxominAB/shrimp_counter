import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load the YOLOv8 model
model = YOLO("best.pt")  # Make sure this model is in the same directory

# Page setup
st.set_page_config(page_title="GROBEST Shrimp Counter", layout="wide")

# Display logo and title
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.jpg", width=100)
with col2:
    st.markdown("<h1 style='font-size: 32px; margin-top: 20px;'>YOLOv8 Object Detection App</h1>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Upload an image to detect objects")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(image_rgb)[0]

    # Draw bounding boxes (without labels)
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    total_objects = len(results.boxes)

    # Show results
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("#### Original Image")
        st.image(image_rgb, channels="RGB", use_column_width=True)

    with col_img2:
        st.markdown("#### Inference Output")
        st.image(image_rgb, channels="RGB", use_column_width=True)
        st.success(f"✅ Total Objects Detected: {total_objects}")