import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image
import os 


# Load the trained model
model = load_model('/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/VGG16/VGG16_train/modelhuasi.h5')

# Load and preprocess images for prediction
folder_path = '/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/VGG16/predictdata'
file_list = os.listdir(folder_path)
img_files = [file for file in file_list if file.endswith((".jpg", ".png", ".jpeg"))]

# Predict the class for each image
for img_file in img_files:
    img = image.load_img(os.path.join(folder_path, img_file), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_indices = np.argsort(predictions[0])[-2:][::-1]  # Get the indices of the top 2 predictions

    # Get the class labels
    class_labels = ['QH', 'YH']  # Replace with your actual class labels

    # Print the predicted class labels
    for index in predicted_indices:
        print("Predicted class:", class_labels[index])
        print("Confidence:", predictions[0][index])

# Delete the files in the detect folder
detect_folder = '/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/yolov5-7.0/runs/detect'
file_list = os.listdir(detect_folder)
for file in file_list:
    file_path = os.path.join(detect_folder, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

detect_folder2 = '/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/VGG16/predictdata'
file_list = os.listdir(detect_folder2)
for file in file_list:
    file_path = os.path.join(detect_folder2, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

detect_folder3 = '/Users/lusicheng/Desktop/summerresearch2024/Visiondetect/yolov5-7.0/data/images'
file_list = os.listdir(detect_folder3)
for file in file_list:
    file_path = os.path.join(detect_folder3, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)


