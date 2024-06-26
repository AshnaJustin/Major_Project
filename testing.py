import os
import numpy as np
import cv2
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from tensorflow.keras.metrics import MeanAbsoluteError # type: ignore

# Register custom objects
custom_objects = {
    'mse': MeanSquaredError(),
    'mae': MeanAbsoluteError()
}

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            images.append(img)
    return images

# Function to load counts from a directory (assuming each line is a bounding box count)
def load_counts_from_folder(folder):
    counts = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(folder, filename)
            with open(txt_path, 'r') as file:
                count = len(file.readlines())  # Counting the number of lines (each representing a bounding box)
                counts.append(count)
    return counts

# Function to use YOLOv8 for object detection (using pre-trained weights)
def detect_people_with_yolov8(frame, model):
    # Make predictions on the frame and return a list of bounding boxes
    results = model(frame)
    boxes = []
    for pred in results.pandas().xywh:
        box = pred[0]
        if box[0] == 0:
            boxes.append(box)
    return boxes

# Function to extract features based on YOLOv8 detections
def extract_features(images, model):
    features = []
    for image in images:
        boxes = detect_people_with_yolov8(image, model)
        feature = len(boxes)  # Count the number of detected persons
        features.append(feature)
    return np.array(features).reshape(-1, 1)  # Reshape for model training

# Load YOLOv8 model
yolo_model = YOLO('yolov8s.pt')  # Load a pre-trained YOLOv8 model

# Load images and labels for testing
image_folder = r"C:\Users\Ashna Justin\Downloads\ash\Images\Test"
label_folder = r"C:\Users\Ashna Justin\Downloads\ash\labels\test_labels"

test_images = load_images_from_folder(image_folder)
test_counts = load_counts_from_folder(label_folder)
print(f'Number of test images: {len(test_images)}')
print(f'Number of test counts: {len(test_counts)}')

# Initialize and load the trained TensorFlow model
model = tf.keras.models.load_model("model_detect.h5", custom_objects=custom_objects)

# Extract features based on YOLOv8 detections for testing images
test_features = extract_features(test_images, yolo_model)

# Convert data to NumPy arrays
X_test_np = np.array(test_features).reshape(-1, 1)
y_test_np = np.array(test_counts)

# Evaluate the model on the testing dataset
loss, mae_value, mse_value = model.evaluate(X_test_np, y_test_np)

print("Mean Absolute Error on Testing Data:", mae_value)
print("Mean Squared Error on Testing Data:", mse_value)

