import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import sys

# Function to download the model from Google Drive
def download_model(file_id, output):
    file_id = '1UxJQk8_uZiYdJPUXmBDy2N7qPJO75ZYx'
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        # Suppress gdown output by redirecting stdout
        with open(os.devnull, 'w') as fnull:
            gdown.download(url, output, quiet=True)
    except Exception as e:
        st.error(f"An error occurred while downloading the model: {e}")
        raise e  # Re-raise the exception to stop execution if download fails

# Define the file ID and output path for the FaceNet model
file_id = '1JLjhjyBCinJxG8Of8_74ukYdIRPWxvmU'  # The actual file ID
output = 'facenet_keras.h5'

# Download the FaceNet model from Google Drive if it doesn't exist locally
if not os.path.exists(output):
    download_model(file_id, output)

# Load the pre-trained FaceNet model
facenet_model = load_model(output)

# Load a pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the image for face recognition model
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)

# Function to load known faces and their embeddings
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    try:
        for person_name in os.listdir('known_faces'):
            person_dir = os.path.join('known_faces', person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    image = cv2.imread(image_path)
                    if image is not None:
                        face_encoding = get_face_encoding(image)
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(person_name)
                    else:
                        st.warning(f"Failed to read image: {image_path}")
    except Exception as e:
        st.error(f"An error occurred while loading known faces: {e}")
        raise e
    return known_face_encodings, known_face_names

# Function to get face encoding using FaceNet
def get_face_encoding(image):
    image = preprocess_image(image)
    embedding = facenet_model.predict(image)[0]
    return embedding

# Function to detect faces in the image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# Function to recognize faces in the image
def recognize_faces(image, known_face_encodings, known_face_names):
    faces = detect_faces(image)
    st.write(f"Detected {len(faces)} faces.")
    recognized_names = []
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        face_encoding = get_face_encoding(face_image)
        similarities = cosine_similarity([face_encoding], known_face_encodings)
        best_match_index = np.argmax(similarities)
        if similarities[0][best_match_index] > 0.5:
            recognized_name = known_face_names[best_match_index]
        else:
            recognized_name = "Unknown"
        recognized_names.append((recognized_name, (x, y, w, h)))
    return recognized_names

# Load known faces and their encodings
known_face_encodings, known_face_names = load_known_faces()

# Title of the Streamlit app
st.title("Face Detection and Recognition App")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a NumPy array
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))  # Ensure the image is in RGB format
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Recognize faces
    recognized_faces = recognize_faces(image_np, known_face_encodings, known_face_names)

    # Draw rectangle around the faces and display names
    for (name, (x, y, w, h)) in recognized_faces:
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image_np, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the processed image
    st.image(image_np, caption='Processed Image', use_column_width=True)

    # List recognized faces
    st.write("Recognized faces:")
    for name, _ in recognized_faces:
        st.write(name)
