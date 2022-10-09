import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import cv2

st.set_page_config(page_title="Melanoma Predictor", page_icon="")
st.markdown("# Melanoma  Predictor")
st.sidebar.markdown("# Melanoma  Predictor")

st.write(
    "Upload an image of the suspected cancerous area. For best results, be sure that the growth is centered, covers a majority of the image, and is minimally obstructed by hair"
)


def get_prediction(img):
    dimension = 224
    channels = 3

    img = img.convert("RGB")
    img = np.asarray(img)
    arr = np.ndarray(shape=(1, dimension, dimension, channels), dtype=np.float32)
    image = img

    size = (dimension, dimension)
    image = cv2.resize(image, size)

    image_array = np.asarray(image)
    arr[0] = (image_array.astype(np.float32) / 127.0) - 1

    cnn = load_model("v4_melanoma")
    return cnn.predict(arr)


file = st.file_uploader("Upload An Image")
cam = 0
if st.button("Take a Picture Instead"):
    del file
    cam = st.camera_input("Take a Picture", disabled=True)
elif file:  # if user uploaded file
    d = st.button("Check if you have melanoma")
    if d:
        img = Image.open(file)
        prediction = get_prediction(img)
        if prediction[0] < 0.5:
            st.write("Melanoma - Confidence: " + str((1 - prediction[0]) * 100) + "%")
        else:
            st.write("Not Melanoma - Confidence: " + str((prediction[0]) * 100) + "%")
if cam:
    d = st.button("Check if you have melanoma")
    if d:
        img = Image.open(file)
        prediction = get_prediction(img)
        if prediction[0] < 0.5:
            st.write("Melanoma - Confidence: " + str((1 - prediction[0]) * 100) + "%")
        else:
            st.write("Not Melanoma - Confidence: " + str((prediction[0]) * 100) + "%")
