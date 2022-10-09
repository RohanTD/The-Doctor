import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Kidney Disease Predictor", page_icon="")
st.markdown("# Kidney Disease  Predictor")
st.sidebar.markdown("# Kidney Disease  Predictor")
st.write("Input your symptoms below")


# ADD THE FORM


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
    # ADD THESE
]

# ADD ENCODING (TEXT -> NUMBERS)

if st.button("Predict"):
    pred = rf.predict(pd.DataFrame([input_arr], columns=symptoms))[0]
    if pred == 1:
        st.write("You likely have kidney disease")
    else:
        st.write("You likely do not have kidney disease")
