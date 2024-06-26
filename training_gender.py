import os
import sys
import numpy as np
import cv2
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import albumentations as A

# Add YOLOv8 repository to the Python path
sys.path.insert(0, 'yolov8')

from ultralytics import YOLO  # Assuming the YOLOv8 library is named 'ultralytics'

# Function to load and augment images from a directory with resizing
def load_and_augment_images_from_folder(folder, img_size=(640, 640)):
    images = []
    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomSizedCrop(min_max_height=(300, 640), height=640, width=640, p=0.5)
    ])
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = augment(image=img)['image']
                images.append(img)
    return images


# Function to load counts from a directory (assuming each line is a bounding box count)
def load_labels_from_folder(folder, label):
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            labels.append(label)
    return labels


# Load YOLOv8 model
def load_yolov8_model(weights_path='yolov8s.pt'):
    model = YOLO(weights_path)
    return model


# Function to use YOLOv8 for object detection (using pre-trained weights)
def detect_people_with_yolov8(frame, model, img_size=640):
    results = model(frame)
    boxes = []
    for result in results:
        if result.boxes is not None:  # Check if there are any detections
            for box in result.boxes:
                if box.cls == 0:  # Assuming class 0 is 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append([x1, y1, x2, y2])
    return boxes


# Function to extract features based on YOLOv8 detections
def extract_features(images, model):
    features = []
    for image in images:
        boxes = detect_people_with_yolov8(image, model)
        feature = len(boxes)  # Example: count of bounding boxes
        features.append(feature)
    return np.array(features).reshape(-1, 1)  # Reshape for model training


# Load images and labels in smaller batches
male_image_folder = r"C:\Users\Ashna Justin\Downloads\Neww\Gender Data\Male"
female_image_folder = r"C:\Users\Ashna Justin\Downloads\Neww\Gender Data\Female"

male_label_folder = r"C:\Users\Ashna Justin\Downloads\Neww\label\male"
female_label_folder = r"C:\Users\Ashna Justin\Downloads\Neww\label\female"

# Assuming batch size of 100
batch_size = 100
male_image_filenames = [f for f in os.listdir(male_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
female_image_filenames = [f for f in os.listdir(female_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

num_batches = (len(male_image_filenames) + len(female_image_filenames)) // batch_size + 1

all_features = []
all_labels = []

model = load_yolov8_model()

for i in range(num_batches):
    batch_image_filenames = male_image_filenames[i * batch_size:(i + 1) * batch_size] + female_image_filenames[i * batch_size:(i + 1) * batch_size]
    batch_labels = [0] * len(male_image_filenames[i * batch_size:(i + 1) * batch_size]) + [1] * len(female_image_filenames[i * batch_size:(i + 1) * batch_size])

    batch_images = [cv2.imread(os.path.join(male_image_folder, fn)) for fn in male_image_filenames[i * batch_size:(i + 1) * batch_size]] + \
                   [cv2.imread(os.path.join(female_image_folder, fn)) for fn in female_image_filenames[i * batch_size:(i + 1) * batch_size]]
    batch_images = [cv2.resize(img, (640, 640)) for img in batch_images if img is not None]
    batch_images = [A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomSizedCrop(min_max_height=(300, 640), height=640, width=640, p=0.5)
    ])(image=img)['image'] for img in batch_images]

    features = extract_features(batch_images, model)
    all_features.extend(features)
    all_labels.extend(batch_labels)

# Convert to NumPy arrays
all_features_np = np.array(all_features).reshape(-1, 1)
all_labels_np = np.array(all_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features_np, all_labels_np, test_size=0.2, random_state=42)

# Model building
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Model Evaluation
model_tf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model_tf.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the trained TensorFlow model
model_tf.save("gender_classification_model.h5")
