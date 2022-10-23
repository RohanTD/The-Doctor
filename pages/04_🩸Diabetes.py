import streamlit as st
import joblib
import pandas as pd
import pickle

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩸")
st.markdown("# Diabetes  Predictor")
st.write("Input your symptoms below")

age = st.number_input("How old are you?", value=25, min_value=25, max_value=70)
gender = st.selectbox("What is your biological gender?", ["Male", "Female"])
col1, col2 = st.columns(2)
weight = col1.checkbox("Have you had sudden weight loss?")
weakness = col1.checkbox("Do you feel weakness?")
polyphagia = col1.checkbox("Do you feel excessively hungry?")
thrush = col1.checkbox(
    "Do you have genital thrush?",
    help="Genital thrush is a yeast infection of the genitals, and symptoms include inflammation, itchiness, and discharge",
)
blurring = col1.checkbox("Is your vision blurred?")
itching = col1.checkbox("Do you have itching?")
polyuria = col1.checkbox(
    "Are you urinating excessively?",
)

irritability = col2.checkbox("Do you feel irritable?")
healing = col2.checkbox("Do you have delayed healing?")
paresis = col2.checkbox(
    "Do you have partial paresis?",
    help="Paresis is the weakning or paralysis of muscles",
)
stiffness = col2.checkbox("Are your muscles stiff?")
alopecia = col2.checkbox(
    "Do you have alopecia?",
    help="Alopecia is a condition of sudden hair loss in patches",
)
obesity = col2.checkbox("Are you obese?")
polydipsia = col2.checkbox("Do you have polydipsia?")

rf = joblib.load("diabetes.joblib")
input_arr = [
    age,
    weight,
    weakness,
    polyphagia,
    thrush,
    blurring,
    itching,
    irritability,
    healing,
    paresis,
    stiffness,
    alopecia,
    obesity,
]
symptoms = [
    "Age",
    "Gender",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]


for i in range(2, len(input_arr)):
    if input_arr[i]:
        input_arr[i] = "Yes"
    else:
        input_arr[i] = "No"

for i in range(len(symptoms)):
    pkl_file = open("diabetes_" + symptoms[i] + ".pkl", "rb")
    lbl = pickle.load(pkl_file)
    pkl_file.close()
    # if symptoms[i] == "Age":
    # st.write(lbl.get_params())
    input_arr[i] = lbl.transform([input_arr[i]])[0]

if st.button("Predict"):
    pred = rf.predict(pd.DataFrame([input_arr], columns=symptoms))[0]
    if pred == 1:
        st.write("You likely have diabetes")
    else:
        st.write("You likely do not have diabetes")
