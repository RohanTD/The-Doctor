from keras.models import load_model
import cv2
from PIL import Image
import numpy as np
import streamlit as st

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
confidence_val = 0.65  # st.slider(
#     "Choose your confidence value. The larger the number, the more false negatives that will result. The lower the number, the more false positives that will result.",
#     value=0.65,
#     max_value=0.98,
#     min_value=0.01,
# )


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

    cnn = load_model("lyme/lyme_model.hdf5")
    return cnn.predict(img)


def getImage(img):

    prediction = get_prediction(img)
    if prediction[0] < (confidence_val * 0.75 + 0.25):
        st.write(
            """<h1 style="text-align: center;"><span style="background-color: #00ff00;"><em>You most likely do not have Erythema Migrans - a primary symptom of Lyme Disease</em></span></h1>
<p><strong><em>If you are experiencing any of the following symptoms of Lyme disease, please contact your doctor:</em></strong></p>
<ol>
<li><em>Muscle Pain</em></li>
<li><em>Bull's Eye Pattern Rash</em></li>
<li><em>Headache</em></li>
<li><em>Stiffness or Swelling</em></li>
</ol>
<h3><em>The following picture is an example of an Erythema Migrans Rash:</em></h3>""",
            unsafe_allow_html=True,
        )
        st.image(
            "lyme/positive_lyme.jpg",
            caption="If you identify a similar abnormality on your skin, please contact your doctor",
        )
    else:
        # st.subheader("Lyme at a Confidence of " + str((prediction[0]) * 100) + "%")
        st.write(
            """<h1 style="text-align: center;"><span style="background-color: #ff6600;"><em>You most likely have Erythema Migrans - a primary symptom of Lyme Disease</em></span></h1>
<p><strong><em>If you are experiencing any of the following symptoms of Lyme disease, please contact your doctor immediately:</em></strong></p>
<ol>
<li><em>Muscle Pain</em></li>
<li><em>Bull's Eye Pattern Rash</em></li>
<li><em>Headache</em></li>
<li><em>Stiffness or Swelling</em></li>
</ol>
<h3><em>The following picture is an example of an Erythema Migrans Rash:</em></h3>""",
            unsafe_allow_html=True,
        )
        st.image(
            "lyme/positive_lyme.jpg",
            caption="If you identify a similar abnormality on your skin, please contact your doctor",
        )


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
footer = """
<style>
footer{
    visibility:visible;
}
footer:before{
    content:"Please keep in mind that this app uses predictors based on machine learning algorithms. Although the results are highly accurate, false positive or negative results can occur. If you still have concerns after consulting our app, please contact your doctor or find a hospital using our locator tool.";
    display:block;
    position:relative;
}
</style>
"""

st.markdown(footer, unsafe_allow_html=True)
