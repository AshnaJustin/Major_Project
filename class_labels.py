import os
import cv2

# Function to convert bounding box coordinates to YOLO format
def convert_to_yolo_format(image_width, image_height, bbox):
    x_center = 0.5  # Center X of the bounding box (always 0.5 for full image)
    y_center = 0.5  # Center Y of the bounding box (always 0.5 for full image)
    width = 1.0     # Width of the bounding box (always 1.0 for full image)
    height = 1.0    # Height of the bounding box (always 1.0 for full image)
    return x_center, y_center, width, height

# Function to annotate images and save YOLO format labels
def annotate_images(image_folder, label_folder):
    class_names = ['male', 'female']
    
    for class_name in class_names:
        class_image_folder = os.path.join(image_folder, class_name)
        class_label_folder = os.path.join(label_folder, class_name)
        os.makedirs(class_label_folder, exist_ok=True)
        
        image_filenames = os.listdir(class_image_folder)
        
        for filename in image_filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_image_folder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                image_height, image_width, _ = img.shape
                yolo_labels = []
                
                # Convert bounding box coordinates to YOLO format (full image annotation)
                x_center, y_center, width, height = convert_to_yolo_format(image_width, image_height, (0, 0, image_width, image_height))
                yolo_labels.append(f"{class_names.index(class_name)} {x_center} {y_center} {width} {height}\n")
                
                # Save YOLO format labels to a .txt file
                label_path = os.path.join(class_label_folder, os.path.splitext(filename)[0] + ".txt")
                with open(label_path, 'w') as file:
                    file.writelines(yolo_labels)

# Example usage
image_folder = r"C:\Users\Ashna Justin\Downloads\gender_detection\test"
label_folder = r"C:\Users\Ashna Justin\Downloads\gender_detection\label_gender"

annotate_images(image_folder, label_folder)
