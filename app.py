import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# App Configuration
st.set_page_config(page_title="Object Detector", layout="wide")

# Load pretrained YOLOv8m model
model = YOLO('best.pt')

# Header Section
col1, col2 = st.columns([1,4])
with col1:
    st.image("images/company_logo.png", width=150)
with col2:
    st.image("images/header_image.jpg", use_column_width=True)

# Main Application
st.title("YOLOv8 Object Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and convert image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Perform detection
    results = model(image)
    
    # Process results
    output_image = image.copy()
    count = 0
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            # Draw bounding boxes without labels
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count += 1
    
    # Display results
    st.image(output_image, channels="BGR", use_column_width=True)
    st.success(f"Total Objects Detected: {count}")