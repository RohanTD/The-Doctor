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
    range = 5  # in miles
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
            radius=range * 1609,
            types=[types.TYPE_HOSPITAL],
        )

        if query_result.has_attributions:
            st.write(query_result.html_attributions)
        if len(query_result.places) == 0:
            st.write(f"There are no hospitals in a {str(range)} mile proximity.")

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

            for j in range(int(place.rating)):
                st.image("star TP.png")
            # st.write(f"Rating: {place.rating}")
            # st.write(f"Phone: {place.formatted_phone_number}")
            st.write(f"Address: {place.formatted_address}")
            st.write(f"Directions: {place.url}")
            st.write(f"Website: {place.website}")
            st.write("\n")


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
