import os
import streamlit as st
import onnxruntime as ort
from PIL import Image
import numpy as np

# Load class names
CLASS_NAMES = sorted(os.listdir("data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"))

# Load ONNX model
onnx_session = ort.InferenceSession("models/plant_disease_model.onnx")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std  
    image = np.transpose(image, (2, 0, 1))  
    image = np.expand_dims(image, axis=0)  
    return image

# Streamlit interface
st.title("Plant Disease Classifier")
st.write("Upload a leaf image to classify the disease.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: img_array})
    prediction = np.argmax(result[0])

    st.write(f"Prediction: {CLASS_NAMES[prediction]}")
