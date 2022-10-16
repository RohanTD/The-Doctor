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

@st.cache(allow_output_mutation=True)
def loadModel(model_name):
    m = load_model(model_name)
    return (m)
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
    cnn = loadModel("v4_melanoma")
    return cnn.predict(arr)


def getImage(img):
    prediction = get_prediction(img)
    if prediction[0] < 0.5:
        st.write("Melanoma - Confidence: " + str((1 - prediction[0]) * 100) + "%")
        st.write("""<h2 style="text-align: center;"><span style="background-color: #ff6600;"><em>You may have cancerous melanoma</em></span></h2>
<h2 style="text-align: left;"><span style="font-size: 14px;">If you notice any of the following symptoms, please get checked by a doctor immediately:</span></h2>
<ol>
<li><strong>Spread of pigment from the border of a spot into the surrounding skin</strong>.</li>
<li><strong>Redness or a new swelling beyond the border of the mole.</strong></li>
<li><strong>Change in sensation, such as itchiness, tenderness, or pain.</strong></li>
<li><strong>Change in the surface of a mole &ndash; scaliness, oozing, bleeding, or the appearance of a lump or bump.</strong></li>
</ol>
<p><em>Every year, approximately 100,000 patients are diagnosed with melanoma and over 7,000 people die of melanoma each year. It is imperative to receive a checkup to avoid future complications in the later stages of possible cancer.</em></p>
<p><strong>The following picture is an image of malignant melanoma:</strong></p>""", unsafe_allow_html = True)
        i = Image.open('Melanoma.jpg')
        st.image(i, caption="If you identify a similar mole on your skin, please contact your doctor")
    else:
        st.write("Not Melanoma - Confidence: " + str((prediction[0]) * 100) + "%")
        st.write("""<h2 style="text-align: center;"><em><span style="background-color: #00ff00;">You most likely do not have melanoma.</span></em></h2>
<p>However, if you notice any of the following symptoms, please get checked by a doctor immediately:</p>
<ol>
<li><strong>Spread of pigment from the border of a spot into the surrounding skin</strong>.</li>
<li><strong>Redness or a new swelling beyond the border of the mole.</strong></li>
<li><strong>Change in sensation, such as itchiness, tenderness, or pain.</strong></li>
<li><strong>Change in the surface of a mole &ndash; scaliness, oozing, bleeding, or the appearance of a lump or bump.</strong></li>
</ol>
<p><strong>The following picture is an image of malignant melanoma:</strong></p>""", unsafe_allow_html = True)
        i = Image.open('Melanoma.jpg')
        st.image(i, caption="If you identify a similar mole on your skin, please contact your doctor")


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
