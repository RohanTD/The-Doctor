from googleplaces import GooglePlaces, types, lang
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import cv2
import requests
import json
from googleplaces import GooglePlaces, types, lang
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import joblib

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


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


def get_hospitals():
    a = "AIzaSyDCd_LRkdU3mHBQ01PY9zSxNat6AI_oD1M"
    range1 = 5  # in miles
    loc_button = Button(label="Allow Location Access")
    loc_button.js_on_event(
        "button_click",
        CustomJS(
            code="""
                navigator.geolocation.getCurrentPosition(
                    (loc) => {
                        document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
                    }
                )
            """
        ),
    )
    response = streamlit_bokeh_events(
        loc_button,
        events="GET_LOCATION",
        key="get_location",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0,
    )
    if response != None:
        places = GooglePlaces(a)
        query_result = places.nearby_search(
            lat_lng={
                "lat": response["GET_LOCATION"]["lat"],
                "lng": response["GET_LOCATION"]["lon"],
            },
            radius=range1 * 1609,
            types=[types.TYPE_HOSPITAL],
        )

        if query_result.has_attributions:
            st.write(query_result.html_attributions)
        if len(query_result.places) == 0:
            st.write(f"There are no hospitals in a {str(range1)} mile proximity.")

        results = []
        for i, place in enumerate(query_result.places):
            place.get_details()

            results.append(
                {
                    "name": place.name,
                    "formatted_address": place.formatted_address,
                    "website": place.website,
                    "gmapsURL": place.url,
                }
            )

            st.subheader(f"{place.name}")
            st.image(
                ["star TP.png"] * (int(place.rating) if place.rating != "" else 0),
                width=20,
            )
            #  for j in range(int(place.rating)):

            # st.write(f"Rating: {place.rating}")
            # st.write(f"Phone: {place.formatted_phone_number}")
            st.write(f"Address: {place.formatted_address}")
            st.write(f"Directions: {place.url}")
            st.write(f"Website: {place.website}")
            st.write("\n")


def covid():
    st.set_page_config(page_title="COVID-19 Predictor ", page_icon="")
    st.markdown("# COVID-19  Predictor")
    col1, col2 = st.columns(2)
    breath = col1.selectbox("Do you have problems breathing?", ["Yes", "No"])
    fever = col1.selectbox("Do you have a fever?", ["Yes", "No"])
    dry_cough = col1.selectbox("Do you have dry cough?", ["Yes", "No"])
    sore_throat = col1.selectbox("Do you have a sore throat?", ["Yes", "No"])
    hypertension = col1.selectbox("Do you have hypertension?", ["Yes", "No"])
    fatigue = col1.selectbox("Do you experience fatigue?", ["Yes", "No"])
    travel = col2.selectbox("Have you traveled abroad recently?", ["Yes", "No"])
    contact = col2.selectbox(
        "Have you had contact with a COVID patient in the last 14 days?", ["Yes", "No"]
    )
    gathering = col2.selectbox(
        "Have you attended a large gathering in the last 14 days?", ["Yes", "No"]
    )
    public = col2.selectbox(
        "Have you visited a public exposed place(EG: Pool) recently?", ["Yes", "No"]
    )
    family = col2.selectbox(
        "Does anyone in your family work in a public exposed place(EG: Hospital) recently?",
        ["Yes", "No"],
    )
    model = joblib.load("covid.pkl")
    input_arr = [
        breath,
        fever,
        dry_cough,
        sore_throat,
        hypertension,
        fatigue,
        travel,
        contact,
        gathering,
        public,
        family,
    ]
    symptoms = [
        "Breathing Problem",
        "Fever",
        "Dry Cough",
        "Sore throat",
        "Hyper Tension",
        "Fatigue ",
        "Abroad travel",
        "Contact with COVID Patient",
        "Attended Large Gathering",
        "Visited Public Exposed Places",
        "Family working in Public Exposed Places",
    ]

    for i in range(len(input_arr)):
        if input_arr[i] == "Yes":
            input_arr[i] = 1
        else:
            input_arr[i] = 0
    inp = pd.DataFrame([input_arr], columns=symptoms)
    y_pred = model.predict(inp)

    if st.button("Predict"):
        if y_pred[0] == 1:
            st.write(
                """<div style="text-align: center;"><span style="font-size: x-large; color: #ff0000;"><strong>You have a HIGH chance of having and/or carrying COVID-19. Please see a doctor immediately.</strong></span></div>
                <div style="text-align: center;">&nbsp;</div>
                <p style="text-align: center;"><strong>&nbsp;</strong><strong style="background-color: #cc99ff;">There have been approximately 553,850,467 cases of COVID-19 worldwide and 6,363,448 total deaths.</strong></p>
                <p style="text-align: center;">&nbsp;</p>
                <p><em>With an incubation period of 14 days, it is important that early diagnosis is conducted, so that individuals can isolate themselves to reduce the chances that they will infect others and allow themselves to seek treatment earlier, likely reducing disease severity, the risk of long-term disability, and death </em></p>
                <h1 style="text-align: center;">&nbsp;</h1>""",
                unsafe_allow_html=True,
            )
    else:
        st.write(
            """<div style="text-align: center;">
            <div><span style="font-size: x-large; color: #ccffcc;"><span style="color: #339966;">You most likely DO NOT have COVID-19. If you are feeling similar symptoms, please get tested.</span></span></div>
            <h1>&nbsp;</h1>
            <p>&nbsp;</p>
            <p><strong>If concerned, there are several ways you can reduce your risk of developing COVID-19, such as:</strong></p>
            <p style="text-align: left;"><span> 1. Giving up and/or avoiding smoking and tobacco.</span></p>
            <p style="text-align: left;"><span> 2. Clean your hands often. Use soap and water, or an alcohol-based hand rub. </span></p>
            <p style="text-align: left;"><span>3. Get vaccinated. Follow local guidance about vaccination.</span></p>
            <p style="text-align: left;"><span> 4. Cover your nose and mouth with your bent elbow or tissue when you cough/sneeze.</span></p>
            <p style="text-align: left;"><span> 5. Stay home if you feel unwell. Open a window if indoors.</span></p>
            <p style="text-align: left;"><span> 6. Wear a mask in public, especially indoors or when physical distancing is not possible.</span></p>
            <p style="text-align: left;"><span>7. Maintain a safe distance from others (at least 1 meter), even if they don&rsquo;t appear to be sick.</span></p>
            <p>&nbsp;</p>
            <p><span style="background-color: #cc99ff;"><strong>There have been approximately 553,850,467 cases of COVID-19 worldwide and 6,363,448 total deaths.</strong></span></p>
            <p>&nbsp;</p>
            <p><em>Luckily there are numerous things we can do to help reduce the spread and avoid serious illness and death. Effective actions like getting the COVID-19 vaccine and a booster shot, wearing a mask, avoiding crowds and indoor places and washing hands often all contribute to fighting the virus.</em></p>
            </div>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    st.title("Melanoma Detection")

    file = st.file_uploader("Upload An Image")
    if file:  # if user uploaded file
        img = Image.open(file)
        prediction = get_prediction(img)
        if prediction[0] < 0.5:
            st.write("Melanoma - Confidence: " + str((1 - prediction[0]) * 100) + "%")
        else:
            st.write("Not Melanoma - Confidence: " + str((prediction[0]) * 100) + "%")

    st.title("Hospital Finder")
    get_hospitals()
