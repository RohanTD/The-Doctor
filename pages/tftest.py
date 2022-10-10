from keras.models import load_model
import cv2
from PIL import Image
import numpy as np
import streamlit as st
model = load_model('lyme.hdf5')

st.set_page_config(page_title="Lyme Disease Predictor", page_icon="")
st.markdown("# Disease  Predictor")

st.write(
    "Upload an image of the suspected cancerous area. For best results, be sure that the growth is centered, well-lit, covers a majority of the image, and is minimally obstructed by hair"
)
# def get_prediction(img):
#     img = cv2.resize(img,(128,128))
#     img = np.reshape(img,[1,128,128,3])
#     cnn = load_model("lyme.hdf5")
#     return cnn.predict(img)

def get_prediction(img):
    dimension = 128
    channels = 3

    img = img.convert("RGB")
    img = np.asarray(img)
    arr = np.ndarray(shape=(1, dimension, dimension, channels), dtype=np.float32)
    image = img

    size = (dimension, dimension)
    image = cv2.resize(image, size)

#     image_array = np.asarray(image)
#     arr[0] = (image_array.astype(np.float32) / 127.0) - 1

    cnn = load_model("lyme.hdf5")
    return cnn.predict(arr)

def getImage(img):
    
    prediction = get_prediction(img)
    if prediction[0] < 0.9:
        st.write("Normal Rash") #- Confidence: " + str((1 - prediction[0]) * 100) + "%")
    else:
        st.write("Lyme - Confidence: " + str((prediction[0]) * 100) + "%")


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
