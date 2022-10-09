import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Kidney Disease Predictor", page_icon="")
st.markdown("# Kidney Disease  Predictor")
st.sidebar.markdown("# Kidney Disease  Predictor")
st.write("Input your symptoms below")


age = st.number_input("How old are you?", value=20)
bp = st.number_input("What is your diastolic blood pressure?", value=90)
sg = st.selectbox(
    "What is your specific gravity?", ["1.005", "1.010", "1.015", "1.020", "1.025"]
)
al = st.number_input(
    "How severe is your albumin level?", value=0, max_value=5, min_value=0
)
su = st.number_input(
    "How severe is your sugar level?", value=0, max_value=5, min_value=0
)
rbc = st.checkbox("Are your red blood cells normal?")
pc = st.checkbox("Is your pus cell count normal?")
pcc = st.checkbox("Are pus cell clumps present?")
ba = st.checkbox("Are bacteria present?")
bgr = st.number_input("What is your blood glucose level?", value=20)
bu = st.number_input("What is your blood urea level?", value=100)
sc = st.number_input("What is your serum creatinine level?", step=0.1)
sod = st.number_input("What is your sodium level?", value=140)
pot = st.number_input("What is your potassium level?", value=4.0, step=0.1)
hemo = st.number_input("What is your hemoglobin level?", value=10.0, step=0.1)
pcv = st.number_input("What is your packed cell volume?", value=40)
wc = st.number_input("What is your white blood cell count?", value=10000, step=100)
rc = st.number_input("What is your red blood cell count?", value=4.0, step=0.1)
htn = st.checkbox("Do you have hypertension?")
dm = st.checkbox("Do you have diabetes mellitus?")
cad = st.checkbox("Do you have coronary artery disease?")
appet = st.selectbox("How is your appetite?", ["Good", "Poor"])
if appet=="Good":
    appet =1
else:
    appet=0
pe = st.checkbox("Do you have pedal edema?")
ane = st.checkbox("Do you have anemia?")


rf = joblib.load("kidney.joblib")
input_arr = [0,
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
    "id",
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

# ADD ENCODING (TEXT -> NUMBERS)

if st.button("Predict"):
    pred = rf.predict(pd.DataFrame([input_arr], columns=symptoms))[0]
    if pred == 1:
        st.write("You likely have kidney disease")
    else:
        st.write("You likely do not have kidney disease")
