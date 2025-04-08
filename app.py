import streamlit as st
import torch
import cv2
print("Using OpenCV from:", cv2.__file__)
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils import draw_bboxes  # Optional: if you want to use your custom bbox drawer

# Load YOLOv8m model
model = YOLO("best.pt")

# Streamlit page configuration
st.set_page_config(page_title="PL Counter Model", layout="wide")

st.title("PL Counter Model")
st.write("Upload an image to detect and count post-larval (PL) shrimp, and calculate average body weight.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # User input for total sample weight
    total_sample_weight = st.number_input("Enter total sample weight (g)", min_value=0.0, step=0.1)

    try:
        # Read and decode image as OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if opencv_image is None:
            raise ValueError("Could not decode the image. Please upload a valid image file.")

        # Run YOLO model
        results = model(opencv_image)

        # Count objects
        num_objects = len(results[0].boxes)

        # Calculate average body weight
        avg_body_weight = total_sample_weight / num_objects if num_objects > 0 else 0.0

        # Draw bounding boxes (either using custom function or inline)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(opencv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Convert image to RGB for display
        result_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

        # Layout: original + result
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Processed Image")
            st.image(result_image, use_container_width=True)
        with col2:
            st.subheader("Detection Summary")
            st.write(f"**Total number of shrimps detected:** {num_objects}")
            st.write(f"**Average body weight:** {avg_body_weight:.2f} g")

    except Exception as e:
        st.error(f"Error processing image: {e}")