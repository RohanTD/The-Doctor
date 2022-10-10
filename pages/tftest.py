from keras.models import load_model
import cv2
import numpy as np
import streamlit as st
model = load_model('lyme.hdf5')

st.set_page_config(page_title="Lyme Disease Predictor", page_icon="")
st.markdown("# Disease  Predictor")

st.write(
    "Upload an image of the suspected cancerous area. For best results, be sure that the growth is centered, well-lit, covers a majority of the image, and is minimally obstructed by hair"
)

def getImage(img):
    img = cv2.resize(img,(128,128))
    img = np.reshape(img,[1,128,128,3])
    classes = model.predict(img)
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
