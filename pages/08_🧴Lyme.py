from keras.models import load_model
import cv2
from PIL import Image
import numpy as np
import streamlit as st

model = load_model("lyme.hdf5")
confidence_val = 0.8
st.set_page_config(page_title="Lyme Disease Predictor", page_icon="🧴")
st.markdown("# Lyme Disease  Predictor")

st.write(
    "Upload an image of the suspected diseased area. For best results, be sure that the area of concern is centered, well-lit, covers a majority of the image, and is minimally obstructed by hair. Please keep in mind, any pictures that are not skin may produce errors or unexpected results."
)
# def get_prediction(img):
#     img = cv2.resize(img,(128,128))
#     img = np.reshape(img,[1,128,128,3])
#     cnn = load_model("lyme.hdf5")
#     return cnn.predict(img)
confidence_val = st.slider(
    "Choose your confidence value. The larger the number, the harder it will be to detect and will produce more false negatives. The lower the number, the algorithm will produce more false positives.",
    value=0.75,
    max_value=0.98,
    min_value=0.01,
)


def get_prediction(img):
    # st.write(type(img))
    dimension = 128
    channels = 3
    img = np.asarray(img)
    # st.write(type(img))
    size = (dimension, dimension)
    img = cv2.resize(img, size)
    img = np.reshape(img, [1, 128, 128, channels])

    #     image_array = np.asarray(image)
    #     arr[0] = (image_array.astype(np.float32) / 127.0) - 1

    cnn = load_model("lyme.hdf5")
    return cnn.predict(img)


def getImage(img):

    prediction = get_prediction(img)
    if prediction[0] < (confidence_val * 0.75 + 0.25):
        st.subheader(
            "Tested Negative for Lyme Disease"
        )  # - Confidence: " + str((1 - prediction[0]) * 100) + "%")
    else:
        st.subheader("Lyme at a Confidence of " + str((prediction[0]) * 100) + "%")


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
