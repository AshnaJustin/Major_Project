import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Path to the test dataset
test_data_dir = r"C:\Users\Ashna Justin\Downloads\gender_detection\test"

# Path to the model saved at a specific epoch
model_path = r'C:\Users\Ashna Justin\Downloads\gender_detection\gender_classification_model.h5'

def evaluate_model(model_path, test_data_dir):
    # Load the saved model
    model = load_model(model_path)

    # Create an ImageDataGenerator for the test data with normalization
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Flow test images in batches of 32 using the test_datagen generator
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),  
        batch_size=32,
        class_mode='binary',    
        shuffle=False          
    )

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_generator)

    print(f'Accuracy on test data: {accuracy * 100:.2f}%')

# Evaluate the model
evaluate_model(model_path, test_data_dir)