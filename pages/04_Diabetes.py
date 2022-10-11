import streamlit as st
import joblib
import pandas as pd
import pickle

st.set_page_config(page_title="Diabetes Predictor", page_icon="")
st.markdown("# Diabetes  Predictor")
st.write("Input your symptoms below")


age = st.number_input("How old are you?", value=25, min_value=25)
gender = st.selectbox("What is your gender?", ["Male", "Female"])
col1, col2 = st.columns(2)
weight = col1.checkbox("Sudden weight loss?")
weakness = col1.checkbox("Weakness?")
polyphagia = col1.checkbox("Polyphagia?")
thrush = col1.checkbox("Genital thrush?")
blurring = col1.checkbox("Blurred vision?")
itching = col2.checkbox("Itching?")
irritability = col2.checkbox("Irritability?")
healing = col2.checkbox("Delayed healing?")
paresis = col2.checkbox("Partial paresis?")
stiffness = col2.checkbox("Muscle stiffness?")
alopecia = col2.checkbox("Alopecia?")
obesity = col2.checkbox("Obesity?")


rf = joblib.load("diabetes.joblib")
input_arr = [
    age,
    gender,
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
    input_arr[i] = lbl.transform([input_arr[i]])[0]

if st.button("Predict"):
    pred = rf.predict(pd.DataFrame([input_arr], columns=symptoms))[0]
    if pred == 1:
        st.write("You likely have diabetes")
    else:
        st.write("You likely do not have diabetes")
