from googleplaces import GooglePlaces, types, lang
import streamlit as st
from googleplaces import GooglePlaces, types, lang
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

st.set_page_config(page_title="Hospital Finder", page_icon="")
st.markdown("# Hospital Finder")


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
