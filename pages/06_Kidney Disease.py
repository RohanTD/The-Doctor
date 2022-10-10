import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Kidney Disease Predictor", page_icon="")
st.markdown("# Kidney Disease  Predictor")
st.sidebar.markdown("# Kidney Disease  Predictor")
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
    sg = None
al = st.number_input(
    "How severe is your albumin level?", value=-1, max_value=5, min_value=-1
)
su = st.number_input(
    "How severe is your sugar level?", value=-1, max_value=5, min_value=-1
)
rbc = st.selectbox("Are your red blood cells normal?", ["Yes", "No", "Don't know"])
if rbc == "Yes":
    rbc = 1
elif rbc == "No":
    rbc = 0
else:
    rbc = None
pc = st.selectbox("Is your pus cell count normal?", ["Yes", "No", "Don't know"])
if pc == "Yes":
    pc = 1
elif pc == "No":
    pc = 0
else:
    pc = None
pcc = st.selectbox("Are pus cell clumps present?", ["Yes", "No", "Don't know"])
if pcc == "Yes":
    pcc = 1
elif pcc == "No":
    pcc = 0
else:
    pcc = None
ba = st.selectbox("Are bacteria present?", ["Yes", "No", "Don't know"])
if ba == "Yes":
    ba = 1
elif ba == "No":
    ba = 0
else:
    ba = None
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
    htn = 1
elif htn == "No":
    htn = 0
else:
    htn = None
dm = st.selectbox("Do you have diabetes mellitus?", ["Yes", "No", "Don't know"])
if dm == "Yes":
    dm = 1
elif dm == "No":
    dm = 0
else:
    dm = None
cad = st.selectbox("Do you have coronary artery disease?", ["Yes", "No", "Don't know"])
if cad == "Yes":
    cad = 1
elif cad == "No":
    cad = 0
else:
    cad = None
appet = st.selectbox("How is your appetite?", ["Good", "Poor", "Don't Know"])
if appet == "Good":
    appet = 1
elif appet == "Poor":
    appet = 0
else:
    appet = None
pe = st.selectbox("Do you have pedal edema?", ["Yes", "No", "Don't know"])
if pe == "Yes":
    pe = 1
elif pe == "No":
    pe = 0
else:
    pe = None
ane = st.selectbox("Do you have anemia?", ["Yes", "No", "Don't know"])
if ane == "Yes":
    ane = 1
elif ane == "No":
    ane = 0
else:
    ane = None


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
            input_arr[i] = None

if st.button("Predict"):
    pred = rf.predict(pd.DataFrame([input_arr], columns=symptoms))[0]
    if pred == 1:
        st.write("You likely have kidney disease")
    else:
        st.write("You likely do not have kidney disease")
