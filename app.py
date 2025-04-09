import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Streamlit Page Configuration
st.set_page_config(
    page_title="GROBEST Shrimp Tools",
    layout="wide",
    page_icon="🦐"
)

# Sidebar Header
st.sidebar.title("🛠️ GROBEST Shrimp Tools")
selected_tool = st.sidebar.selectbox(
    "Select a Tool",
    ["PL counter"]
)

# Main App Header
st.title("🦐 GROBEST Shrimp Tools Dashboard")

# PL Counter Tool
if selected_tool == "PL counter":
    st.markdown("### 📸 Upload an Image for Postlarvae (PL) Counting")

    # Load the YOLOv8 model
    model_path = "best.pt"  # Replace with your actual path to trained YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        st.stop()

    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert uploaded image to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Run inference
        with st.spinner("🔍 Counting Postlarvae..."):
            results = model(opencv_image)
            num_objects = len(results[0].boxes)

            # Draw bounding boxes
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(
                    opencv_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

            # Convert image from BGR to RGB for display
            result_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(result_image, caption="Detected PL", use_column_width=True)
        with col2:
            st.metric(label="🔢 Total PL Count", value=num_objects)