import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# Load YOLOv8 model
yolo_model = YOLO('yolov8s.pt')

# Load Gender Classification model (Ensure you have a pre-trained model)
gender_model = load_model('gender_classification_model.h5')

# Function to detect people and return bounding boxes
def detect_people_with_yolov8(frame):
    results = yolo_model(frame)
    boxes = results[0].boxes
    return boxes

# Function to classify gender
def classify_gender(cropped_face):
    cropped_face = cv2.resize(cropped_face, (224, 224))  # Resize to match model input size
    cropped_face = np.expand_dims(cropped_face, axis=0)  # Expand dims to match model input shape
    cropped_face = cropped_face / 255.0  # Normalize
    prediction = gender_model.predict(cropped_face)
    return 'Male' if prediction[0][0] > 0.5 else 'Female'

# Initialize Tkinter window
window = tk.Tk()
window.title("Real-Time Detection")

# Set up the video capture
cap = cv2.VideoCapture(0)
alert_threshold = 5  # Example threshold for alert

# Set up a label to display the video feed
label = Label(window)
label.pack()

# Initialize counters and tracker
current_count = 0
max_people_count = 0
next_person_id = 0
people_tracker = {}  # To track people's centroids

def calculate_centroid(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
    return (x1 + x2) // 2, (y1 + y2) // 2

def assign_unique_id(centroid):
    global next_person_id
    for person_id, tracked_centroid in people_tracker.items():
        if np.linalg.norm(np.array(centroid) - np.array(tracked_centroid)) < 50:  # Threshold distance
            people_tracker[person_id] = centroid
            return person_id
    # Assign new ID if no existing person is close enough
    people_tracker[next_person_id] = centroid
    next_person_id += 1
    return next_person_id - 1

# Function to send alert
def send_alert(count):
    print(f"Alert! {count} people detected!")

def detect_and_display():
    global current_count, max_people_count

    ret, frame = cap.read()
    if not ret:
        return
    
    boxes = detect_people_with_yolov8(frame)
    centroids = [calculate_centroid(box) for box in boxes if box.cls == 0]
    current_ids = [assign_unique_id(centroid) for centroid in centroids]

    current_count = len(current_ids)
    
    # Update max people count if the current count exceeds it
    if current_count > max_people_count:
        max_people_count = current_count

    # Draw bounding boxes, unique IDs, and gender
    for box, person_id in zip(boxes, current_ids):
        if box.cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            cropped_face = frame[y1:y2, x1:x2]
            gender = classify_gender(cropped_face)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f'Person {person_id} ({gender})'
            (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if current_count > alert_threshold:
        send_alert(current_count)

    # Display current count on the frame
    cv2.putText(frame, f"Current Count: {current_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame to ImageTk format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    
    # Update the label with the new frame
    label.config(image=img_tk)
    label.image = img_tk
    
    # Schedule the next update
    label.after(10, detect_and_display)

# Start the detection loop
detect_and_display()

# Start the Tkinter event loop
window.mainloop()

# Release the video capture on close
cap.release()
