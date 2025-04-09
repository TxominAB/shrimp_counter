import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the pretrained YOLOv8 model from the local directory
model = YOLO('best.pt')

# Function to perform inference
def perform_inference(image):
    results = model.predict(image)
    return results

# Streamlit app
st.title("Object Detection with YOLOv8")
st.image("company_logo.jpg", use_column_width=True)
st.image("header_image.jpg", use_column_width=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Perform inference
    results = perform_inference(image_cv)
    
    # Draw bounding boxes
    for box in results['boxes']:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display results
    st.image(image_cv, caption="Detected Objects", use_column_width=True)
    st.write(f"Total objects detected: {len(results['boxes'])}")