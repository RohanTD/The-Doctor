import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import cv2

st.set_page_config(page_title="Melanoma Predictor", page_icon="👦🏻")
st.markdown("# Melanoma  Predictor")

st.write(
    "Upload an image of the suspected cancerous area. For best results, be sure that the growth is centered, well-lit, covers a majority of the image, and is minimally obstructed by hair"
)

@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
def load_model():
	  return load_model("v4_melanoma")
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

    cnn = load_model()
    return cnn.predict(arr)


def getImage(img):
    prediction = get_prediction(img)
    if prediction[0] < 0.5:
        st.write("Melanoma - Confidence: " + str((1 - prediction[0]) * 100) + "%")
    else:
        st.write("Not Melanoma - Confidence: " + str((prediction[0]) * 100) + "%")


fileHold = st.empty()
camHold = st.empty()


def fileMethod():
    fileHold.subheader("Upload")
    file2 = fileHold.file_uploader("Upload an image")
    if file2:  # if user uploaded file
        camHold.empty()
        getImage(Image.open(file2))


def camMethod():
    camHold.subheader("Take photo")
    cam = camHold.camera_input("Take a photo (please allow webcam access)")
    if cam is not None:
        fileHold.empty()
        getImage(Image.open(cam))


fileMethod()
camMethod()


# if st.button(" Take a picture instead"):
#     placeholder.empty()
