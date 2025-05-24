import base64
import numpy as np
import cv2
from PIL import Image


def base64_to_image(base64_string: str):
    image_data = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def preprocess_for_model(image_cv):
    # Convert BGR (OpenCV) to RGB (PIL)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)
