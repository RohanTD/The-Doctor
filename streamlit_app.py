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
    range = 10000  # in miles
    # r = requests.post(f"https://www.googleapis.com/geolocation/v1/geolocate?key={a}")
    apiKey = "9fe19182c5bf4d1bb105da08e593a578"
    # print(len(County))
    US_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    confirmed = pd.read_csv(US_confirmed)
    FIPSs = (
        confirmed.groupby(["Province_State", "Admin2"])
        .FIPS.unique()
        .apply(pd.Series)
        .reset_index()
    )
    FIPSs.columns = ["State", "County", "FIPS"]
    FIPSs["FIPS"].fillna(0, inplace=True)
    FIPSs["FIPS"] = FIPSs.FIPS.astype(int).astype(str).str.zfill(5)
    CA_counties = (
        confirmed[confirmed.Province_State == "California"].Admin2.unique().tolist()
    )

    County = st.multiselect("Select counties", CA_counties, default=["Yolo"])
    f = FIPSs[FIPSs.County == County[0]].FIPS.values[0]
    # print(f)
    path1 = (
        "https://data.covidactnow.org/latest/us/counties/"
        + f
        + ".OBSERVED_INTERVENTION.timeseries.json?apiKey="
        + apiKey
    )

    r = requests.post(path1)
    st.write(r)
    response = json.loads(r.content)
    st.write(response)

    st.write(f'(Location accurate to {response["accuracy"]/1609} miles.)\n')
    places = GooglePlaces(a)
    query_result = places.nearby_search(
        lat_lng={
            "lat": response["location"]["lat"],
            "lng": response["location"]["lng"],
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

    st.write(f"({i}) : {place.name}")
    st.write(f"  -  {place.formatted_address}")
    st.write(f"  -  {place.url}")
    st.write(f"  -  {place.website}")


if __name__ == "__main__":
    st.title("Welcome To the Medical App!")

    file = st.file_uploader("Upload An Image (Melanoma Detection)")
    if file:  # if user uploaded file
        img = Image.open(file)
        prediction = get_prediction(img)
        if prediction[0] < 0.5:
            st.write("Melanoma - Confidence: " + str((1 - prediction[0]) * 100) + "%")
        else:
            st.write("Not Melanoma - Confidence: " + str((prediction[0]) * 100) + "%")
    get_hospitals()
