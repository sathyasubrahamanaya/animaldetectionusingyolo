import cv2
import numpy as np
from ultralytics import YOLO

# Set a standard image size (same as training)
IMAGE_SIZE = 640

def load_model(model_path="animal_detection_model.pt"):
    """
    Loads the YOLO model from the given path.
    Change model_path if your saved weights are stored elsewhere.
    """
    model = YOLO(model_path)
    return model

def preprocess_image(uploaded_file):
    """
    Reads an uploaded image file from Streamlit, decodes it, and resizes it to IMAGE_SIZE.
    Returns both the original image (for display) and the resized image (for inference).
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Resize image to fixed dimensions (without preserving aspect ratio, as done during training)
    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image, image_resized

def convert_bgr_to_rgb(image):
    """
    Converts an image from BGR to RGB.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_rgb_to_lab(image):
    """
    Converts an image from RGB to LAB color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def convert_lab_to_rgb(image):
    """
    Converts an image from LAB to RGB color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

def run_inference(model, image):
    """
    Runs inference on the preprocessed image using the loaded YOLO model.
    The image should be in the format (IMAGE_SIZE, IMAGE_SIZE, 3) in BGR.
    """
    results = model(image)
    return results

def get_result_image(results):
    """
    Extracts and returns the image with bounding boxes drawn from the inference results.
    The output image is in BGR color space.
    """
    result_img = results[0].plot()
    return result_img
