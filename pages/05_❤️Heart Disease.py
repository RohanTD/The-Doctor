import streamlit as st
import joblib
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.markdown("# Heart Disease  Predictor")
st.write("Input your symptoms below")

col1, col2 = st.columns(2)
age = col1.number_input("How old are you?", value=29, min_value=29)  # FIX
trestbps = col1.number_input(
    "What is your resting systolic blood pressure (mmHg)?", value=120  # FIX
)
chol = col1.number_input("What is your cholesterol level (mg/dL)?", value=250)  # FIX
thalach = col1.number_input(
    "What was your maximum heart rate achieved?", value=150  # FIX
)
ca = col1.number_input(
    "How many major vessels were colored by fluoroscopy?",
    value=1,
    max_value=3,
    min_value=0,
)

sex = col2.selectbox("What is your gender?", ["Male", "Female"])
slope = col2.selectbox(
    "What was the slope of the peak exercise ST segment (ECG)?",
    ["Upsloping", "Flat", "Downsloping"],
)

thal = col2.selectbox(
    "Which type of thalassemia applies?",
    ["Normal", "Fixed Defect", "Reversible Defect"],
)

cp = col2.selectbox(
    "What type of chest pain are you having?",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptompatic"],
)
restecg = col2.selectbox(
    "What were your resting ECG results?",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy by Estes' Criteria",
    ],
)

fbs = col1.checkbox("Is your fasting blood sugar more than 120 mg/dL?")
exang = col1.checkbox("Do you experience exercise-induced angina?")
oldpeak = col2.checkbox(
    "Do you have an ST depression induced by exercise relative to rest (ECG)?"
)


if sex == "Male":
    sex = 1
else:
    sex = 0

if cp == "Typical Angina":
    cp = 0
elif cp == "Atypical Angina":
    cp = 1
elif cp == "Non-Anginal Pain":
    cp = 2
else:
    cp = 3

if restecg == "Normal":
    restecg = 0
elif restecg == "ST-T Wave Abnormality":
    restecg = 1
else:
    restecg = 2

if slope == "Upsloping":
    slope = 0
elif slope == "Flat":
    slope = 1
else:
    slope = 2

if thal == "Normal":
    thal = 1
elif thal == "Fixed Defect":
    thal = 2
else:
    thal = 3

rf = joblib.load("heart.joblib")
input_arr = [
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal,
]
symptoms = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
for i in [5, 8, 9]:
    if input_arr[i]:
        input_arr[i] = 1
    else:
        input_arr[i] = 0

for i in range(len(symptoms)):
    pkl_file = open("heart_" + symptoms[i] + ".pkl", "rb")
    lbl = pickle.load(pkl_file)
    pkl_file.close()
    input_arr[i] = lbl.transform([input_arr[i]])[0]

if st.button("Predict"):
    pred = rf.predict(pd.DataFrame([input_arr], columns=symptoms))[0]
    if pred == 1:
        st.write("You likely have heart disease")
    else:
        st.write("You likely do not have heart disease")
