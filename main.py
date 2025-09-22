import cv2
import os
import numpy as np


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Syntaxen er from package.file import class/function
from logic.face_detector import DetectionVisualizer


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Denne funktion opretter en mappe, hvis den ikke findes
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    print_hi('PyCharm')

    # Configuration
    IMAGE_FILE_NAME = "brother-977170_1280.jpg"

    IMAGE_FILE = os.path.join("images", IMAGE_FILE_NAME)

    MODEL_FILE = os.path.join("models", "blaze_face_short_range.tflite")
    OUT_DIR = "out"
    OUT_FILE = os.path.join(OUT_DIR, IMAGE_FILE_NAME+"_annotated.jpg")

    if not os.path.isfile(IMAGE_FILE):
        raise FileNotFoundError(f"Image not found: {IMAGE_FILE}")

    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Place a MediaPipe face detection .tflite model at this path."
        )

    ensure_dir(OUT_DIR)

    # Create detector and config

    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)

    # "The minimum confidence score for the face detection to be considered successful."
    # Default er 0.5
    detection_confidence = 0.5

    # Default er image men vi kan også bruge VIDEO eller LIVE_STREAM
    running_mode = vision.RunningMode.IMAGE

    # "The minimum non-maximum-suppression threshold for face detection to be considered overlapped."
    # Default er 0.3
    min_suppression_threshold = 0.3


    options = vision.FaceDetectorOptions(base_options=base_options,
                                         min_detection_confidence=detection_confidence,
                                         running_mode=running_mode,
                                         min_suppression_threshold=min_suppression_threshold)


    detector = vision.FaceDetector.create_from_options(options)

    # Load into mediapipe
    mp_image = mp.Image.create_from_file(IMAGE_FILE)

    # Run detection
    detection_result = detector.detect(mp_image)

    print(f"Found {len(detection_result.detections)} faces")
    if len(detection_result.detections) == 0:
        print("No faces found, exiting")
        exit(0)

    # Visualize detection result
    # mp_image.numpy_view() is an RGB numpy array, vi skal bruge rgb til visualizer.visualize
    image_rgb = np.copy(mp_image.numpy_view())

    visualizer = DetectionVisualizer()
    annotated_rgb = visualizer.visualize(image_rgb, detection_result)

    # Save to disk
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(OUT_FILE, annotated_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write output to {OUT_FILE}")

    print(f" Annotated image saved to: {OUT_FILE}")


