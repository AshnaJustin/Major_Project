import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from tensorflow.keras.metrics import MeanAbsoluteError  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from PIL import Image, ImageTk
import pickle
import argparse
import time
import os

# Register custom objects for keras metrics
custom_objects = {
    'MeanSquaredError': MeanSquaredError,
    'MeanAbsoluteError': MeanAbsoluteError,
    'mae': MeanAbsoluteError  # Explicitly register 'mae'
}

# Load the trained TensorFlow models with custom object scope
gender_model = tf.keras.models.load_model("gender_classification_model.h5", custom_objects=custom_objects)
additional_model = tf.keras.models.load_model("model_detect.h5", custom_objects=custom_objects)

# Print model summaries to inspect input shapes
print("Gender Model Summary:")
print(gender_model.summary())

print("Additional Model Summary:")
print(additional_model.summary())

# Load the YOLOv8 model
yolo_model = YOLO('yolov8s.pt')

# Function to use YOLOv8 for object detection
def detect_people_with_yolov8(frame):
    results = yolo_model(frame)
    boxes = results[0].boxes  # Extract the bounding boxes
    return boxes

# Function to classify gender
def classify_gender(image, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
    person_img = image[y1:y2, x1:x2]
    person_img_resized = cv2.resize(person_img, (224, 224))  # Resize to match model input size
    person_img_array = np.expand_dims(person_img_resized, axis=0) / 255.0  # Preprocess the image
    gender_prediction = gender_model.predict(person_img_array)
    gender = 'Male' if gender_prediction[0][0] > 0.5 else 'Female'
    return gender

# Function to apply additional classification
def classify_additional(feature):
    feature_array = np.array([[feature]])  # Reshape for model input if necessary
    additional_prediction = additional_model.predict(feature_array)
    category = 'Adult' if additional_prediction[0][0] > 0.5 else 'Child'
    return category

# Function to process the video and generate heatmap
def process_video_with_heatmap(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    max_people_count = 0
    heatmap_accumulator = np.zeros((height, width), dtype=np.float32)
    
    frame_data = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        boxes = detect_people_with_yolov8(frame)
        people_count = sum(1 for box in boxes if box.cls == 0)

        # Update the maximum count if the current frame's count is higher
        max_people_count = max(max_people_count, people_count)

        # Draw bounding boxes and accumulate heatmap data
        activity_labels = []
        for idx, box in enumerate(boxes, start=1):
            if box.cls == 0:  # Ensure the detected object is a person
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                gender = classify_gender(frame, box)
                # Use number of detected persons as a feature for additional classification
                category = classify_additional(people_count)
                label = f'Person {idx} ({gender}, {category})'
                activity_labels.append(label)
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Accumulate the heatmap
                heatmap_accumulator[y1:y2, x1:x2] += 1

        # Display maximum count
        frame_count_label = f'Frame Count: {people_count}'
        total_count_label = f'Total Count: {max_people_count}'
        cv2.putText(frame, frame_count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, total_count_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_data.append([cap.get(cv2.CAP_PROP_POS_FRAMES), people_count, max_people_count, timestamp, activity_labels])

        out.write(frame)

        # Display the frame with counts
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print the maximum count after processing all frames
    print(f'Total people detected in the video: {max_people_count}')

    # Plot and save the final heatmap
    plt.figure()
    sns.heatmap(heatmap_accumulator, cmap='viridis')
    plt.title('Heatmap of People Detection')
    heatmap_path = 'heatmap.png'
    plt.savefig(heatmap_path)
    plt.show()
    
    end_time = time.time()
    fps_processed = len(frame_data) / (end_time - start_time)
    print(f"Processed FPS: {fps_processed}")

    return heatmap_path, frame_data

# Function to display heatmap using Tkinter
def display_heatmap_tkinter(image_path):
    root = tk.Tk()
    root.title("Heatmap")

    img = Image.open(image_path)
    img = ImageTk.PhotoImage(img)

    panel = tk.Label(root, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")

    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='People Counting and Gender Classification in Video')
    parser.add_argument('--input', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output video file')
    parser.add_argument('--heatmap', type=str, required=True, help='Path to the output heatmap image')
    args = parser.parse_args()

    # Process video and save frame data for the dashboard and gender recognition script
    heatmap_path, frame_data = process_video_with_heatmap(args.input, args.output)

    with open('frame_data.pkl', 'wb') as f:
        pickle.dump(frame_data, f)

    # Display the generated heatmap using Tkinter
    display_heatmap_tkinter(heatmap_path)
