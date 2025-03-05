import streamlit as st
import cv2
import numpy as np
from model import (
    load_model, 
    preprocess_image, 
    convert_bgr_to_rgb, 
    run_inference, 
    get_result_image,
    convert_rgb_to_lab,  # Optional: if you need LAB conversion for any processing
    convert_lab_to_rgb   # Optional: if needed for reconversion
)

st.title("Animal Detection with YOLO")
st.write("Upload an image to detect animals with bounding boxes.")

# Load the YOLO model (this uses caching to avoid reloading on every run)
@st.cache_resource()
def get_model():
    return load_model()

model = get_model()

# File uploader accepts JPG, JPEG, PNG files
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image: get both the original and resized image
    original_image, preprocessed_image = preprocess_image(uploaded_file)
    
    # (Optional) If you want to experiment with color space conversions,
    # you can convert the image from RGB to LAB and back:
    # lab_image = convert_rgb_to_lab(convert_bgr_to_rgb(original_image))
    # original_image = convert_bgr_to_rgb(convert_lab_to_rgb(lab_image))
    
    # Display the original image (convert BGR to RGB for correct colors)
    st.image(convert_bgr_to_rgb(original_image), caption="Original Image", use_column_width=True)
    
    if st.button("Detect Animals"):
        with st.spinner("Running Inference..."):
            # Run the model on the preprocessed image (which is resized to 640x640)
            results = run_inference(model, preprocessed_image)
            # Retrieve the result image with bounding boxes (still in BGR)
            result_image = get_result_image(results)
            # Convert the result image to RGB before displaying
            result_image_rgb = convert_bgr_to_rgb(result_image)
            st.image(result_image_rgb, caption="Detection Result", use_column_width=True)
