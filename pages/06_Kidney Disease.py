import streamlit as st
from cmath import nan
import joblib
import pandas as pd
import pickle

st.set_page_config(page_title="Kidney Disease Predictor", page_icon="")
st.markdown("# Kidney Disease  Predictor")
st.write(
    "Input your symptoms below - if you do not know the answer to any numerical input question, please input a value of -1"
)


age = st.number_input("How old are you?", value=20, min_value=10)
bp = st.number_input("What is your diastolic blood pressure?", value=-1)
sg = st.selectbox(
    "What is your specific gravity?",
    [1.005, 1.010, 1.015, 1.020, 1.025, "Don't know"],
)
if sg == "Don't know":
    sg = nan
al = st.number_input(
    "How severe is your albumin level?", value=-1, max_value=5, min_value=-1
)
su = st.number_input(
    "How severe is your sugar level?", value=-1, max_value=5, min_value=-1
)
rbc = st.selectbox("Are your red blood cells normal?", ["Yes", "No", "Don't know"])
if rbc == "Yes":
    rbc = "normal"
elif rbc == "No":
    rbc = "abnormal"
else:
    rbc = nan
pc = st.selectbox("Is your pus cell count normal?", ["Yes", "No", "Don't know"])
if pc == "Yes":
    pc = "normal"
elif pc == "No":
    pc = "abnormal"
else:
    pc = nan
pcc = st.selectbox("Are pus cell clumps present?", ["Yes", "No"])
if pcc == "Yes":
    pcc = "present"
elif pcc == "No":
    pcc = "notpresent"
ba = st.selectbox("Are bacteria present?", ["Yes", "No"])
if ba == "present":
    ba = 1
elif ba == "notpresent":
    ba = 0
bgr = st.number_input("What is your blood glucose level?", value=-1, min_value=-1)
bu = st.number_input("What is your blood urea level?", value=-1, min_value=-1)
sc = st.number_input("What is your serum creatinine level?", step=0.1, min_value=-1.0)
sod = st.number_input("What is your sodium level?", value=-1, min_value=-1)
pot = st.number_input(
    "What is your potassium level?", value=4.0, step=0.1, min_value=-1.0
)
hemo = st.number_input(
    "What is your hemoglobin level?", value=10.0, step=0.1, min_value=-1.0
)
pcv = st.number_input("What is your packed cell volume?", value=-1, min_value=-1)
wc = st.number_input(
    "What is your white blood cell count?", value=10000, step=100, min_value=-1
)
rc = st.number_input(
    "What is your red blood cell count?", value=4.0, step=0.1, min_value=-1.0
)
htn = st.selectbox("Do you have hypertension?", ["Yes", "No", "Don't know"])
if htn == "Yes":
    htn = "yes"
elif htn == "No":
    htn = "no"
else:
    htn = nan
dm = st.selectbox("Do you have diabetes mellitus?", ["Yes", "No", "Don't know"])
if dm == "Yes":
    dm = "yes"
elif dm == "No":
    dm = "no"
else:
    dm = nan
cad = st.selectbox("Do you have coronary artery disease?", ["Yes", "No", "Don't know"])
if cad == "Yes":
    cad = "yes"
elif cad == "No":
    cad = "no"
else:
    cad = nan
appet = st.selectbox("How is your appetite?", ["Good", "Poor", "Don't Know"])
if appet == "Good":
    appet = "good"
elif appet == "Poor":
    appet = "poor"
else:
    appet = nan
pe = st.selectbox("Do you have pedal edema?", ["Yes", "No", "Don't know"])
if pe == "Yes":
    pe = "yes"
elif pe == "No":
    pe = "no"
else:
    pe = nan
ane = st.selectbox("Do you have anemia?", ["Yes", "No", "Don't know"])
if ane == "Yes":
    ane = "yes"
elif ane == "No":
    ane = "no"
else:
    ane = nan


rf = joblib.load("kidney.joblib")
input_arr = [
    age,
    bp,
    sg,
    al,
    su,
    rbc,
    pc,
    pcc,
    ba,
    bgr,
    bu,
    sc,
    sod,
    pot,
    hemo,
    pcv,
    wc,
    rc,
    htn,
    dm,
    cad,
    appet,
    pe,
    ane,
]
symptoms = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]

for i in range(len(input_arr)):
    if i == 1 or 3 or 4 or 9 or 10 or 11 or 12 or 13 or 14 or 15 or 16 or 17:
        if input_arr[i] == -1:
            input_arr[i] = nan
        else:
            input_arr[i] = float(input_arr[i])

for i in range(len(symptoms)):
    pkl_file = open("kidney_" + symptoms[i] + ".pkl", "rb")
    lbl = pickle.load(pkl_file)
    pkl_file.close()
    input_arr[i] = lbl.transform([input_arr[i]])[0]

if st.button("Predict"):
    pred = rf.predict(pd.DataFrame([input_arr], columns=symptoms))[0]
    if pred == 1:
        st.write("You likely have kidney disease")
    else:
        st.write("You likely do not have kidney disease")
