import cv2
from deepface import DeepFace

def detect_emotion(image_path):
    try:
        # Analyze image using DeepFace with 'opencv' backend (no TensorFlow)
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], detector_backend='opencv')
        return result[0]['dominant_emotion']
    except Exception as e:
        return f"Error: {str(e)}"
