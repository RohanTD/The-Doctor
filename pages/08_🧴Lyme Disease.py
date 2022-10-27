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
    "Upload an image of the suspected diseased area. For best results, be sure that the area of concern is centered, well-lit, covering a majority of the image, and minimally obstructed by hair. Please keep in mind, any pictures that are not skin may produce errors or unexpected results."
)
# def get_prediction(img):
#     img = cv2.resize(img,(128,128))
#     img = np.reshape(img,[1,128,128,3])
#     cnn = load_model("lyme.hdf5")
#     return cnn.predict(img)
confidence_val = st.slider(
    "Choose your confidence value. The larger the number, the more false negatives that will result. The lower the number, the more false positives that will result.",
    value=0.65,
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
        st.write("""<h1 style="text-align:center"><span style="font-family:Comic Sans MS, cursive"><em><span style="background-color:#2ecc71">You most likely do not have Erythema Migrans - a primary symptom of Lyme Disease</span></em></span></h1>

<p><strong><span style="font-family:Comic Sans MS, cursive"><em>If you are experiencing any of the following symptoms of Lyme disease, please contact your doctor:</em></span></strong></p>

<ol>
	<li><em><span style="font-family:Comic Sans MS,cursive">Muscle Pain</span></em></li>
	<li><em><span style="font-family:Comic Sans MS,cursive">Bull&#39;s Eye Pattern Rash</span></em></li>
	<li><em><span style="font-family:Comic Sans MS,cursive">Headache</span></em></li>
	<li><em><span style="font-family:Comic Sans MS,cursive">Stiffness or Swelling</span></em></li>
</ol>

<h3 style="text-align:center"><em><span style="font-family:Comic Sans MS,cursive">Please keep in mind that this is not an official diagnosis, we are doing the best we can with the pictures provided to use from the internet. The following picture is an example of an Erythema Migrans Rash:</span></em></h3>""", unsafe_allow_html = True)
	st.image("EM.jpg")
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
