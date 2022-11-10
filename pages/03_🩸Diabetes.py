import streamlit as st
import joblib
import pandas as pd
import pickle

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩸")
st.markdown("# Diabetes  Predictor")
st.write("Input your symptoms below")

age = st.number_input("How old are you?", value=25, min_value=25, max_value=70)
col1, col2 = st.columns(2)

polyuria = col1.checkbox(
    "Are you urinating excessively?",
)

polyphagia = col1.checkbox("Do you feel excessively hungry?")
polydipsia = col1.checkbox("Do you feel excessively thirsty?")
weight = col1.checkbox("Have you lost weight suddenly?")
weakness = col1.checkbox("Do you feel weakness?")
thrush = col1.checkbox(
    "Do you have genital thrush?",
    help="Genital thrush is a yeast infection of the genitals, and symptoms include inflammation, itchiness, and discharge",
)
blurring = col1.checkbox("Is your vision blurred?")
itching = col2.checkbox("Do you have itching?")

irritability = col2.checkbox("Do you feel irritable?")
healing = col2.checkbox("Do you have delayed healing?")
paresis = col2.checkbox(
    "Do you have partial paresis?",
    help="Paresis is the weakening or paralysis of muscles",
)
stiffness = col2.checkbox("Are your muscles stiff?")
alopecia = col2.checkbox(
    "Do you have alopecia?",
    help="Alopecia is a condition of sudden hair loss in patches",
)
obesity = col2.checkbox("Are you obese?")

rf = joblib.load("diabetes/diabetes.joblib")
input_arr = [
    age,
    polyuria,
    polydipsia,
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
    "Polyuria",
    "Polydipsia",
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


for i in range(1, len(input_arr)):
    if input_arr[i]:
        input_arr[i] = "Yes"
    else:
        input_arr[i] = "No"

for i in range(len(symptoms)):
    pkl_file = open("diabetes/encodings/diabetes_" + symptoms[i] + ".pkl", "rb")
    lbl = pickle.load(pkl_file)
    pkl_file.close()
    # if symptoms[i] == "Age":
    # st.write(lbl.get_params())
    input_arr[i] = lbl.transform([input_arr[i]])[0]

if st.button("Predict"):
    pred = rf.predict(pd.DataFrame([input_arr], columns=symptoms))[0]
    if pred == 1:
        st.write(
            """<div style="text-align: center;">
                <div><span style="font-size: x-large; background-color: #ff6600;">You have a HIGH chance of having diabetes or being prediabetic. Please see a doctor immediately.</span></div>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p><strong>Diabetes is leading cause of death across the world. An estimated 6.7 million people die from diabetes yearly, representing 11.3% of all global deaths.</strong></p>
                <p>&nbsp;</p>
                <p><em>Identifying those at the highest risk of diabetes, diagnosing as early as possible, and ensuring patients receive appropriate treatment at the correct time can prevent premature and consequential deaths. Access to noncommunicable disease medicines and basic health technologies is essential to ensure that those in need receive appropriate care.</em></p>
                </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.write(
            """<div style="text-align: center;"><span style="font-size: x-large; background-color: #00ff00;">You most likely DO NOT have diabetes.</span></div>
                <p><strong>If concerned, there are several ways you can reduce your risk of developing diabetes, such as:</strong></p>
                <p><span>1. Lowering your blood and cholesterol levels.</span></p>
                <p><span> 2. Eating a healthy, balanced diet which includes plant based foods.</span></p>
                <p><span> 3. Maintaining a healthy weight.</span></p>
                <p><span> 4. Giving up and/or avoiding smoking and tobacco.</span></p>
                <p><span> 5. Reducing alcohol consumption.</span></p>
                <p><span> 6. Keeping blood pressure under control.</span></p>
                <p><span> 7. Being consistently active and involved in physical activity.</span></p>
                <p>&nbsp;</p>
                <p><strong>Diabetes is a leading cause of death globally. An estimated 6.7 million people died from cardiovascular diseases per year, representing 11.2% of all global deaths.</strong></p>
                <p>&nbsp;</p>
                <p><em>Identifying those at the highest risk of Diabetes early on,diagnosing as early as possible, and ensuring patients receive appropriate treatment at the correct time can prevent premature and consequential deaths. Access to noncommunicable disease medicines and basic health technologies is essential to ensure that those in need receive appropriate care.</em></p>
                """,
            unsafe_allow_html=True,
        )
footer = """
<style>
footer{
    visibility:visible;
}
footer:before{
    content:"Please keep in mind that this app uses predictors based on machine learning algorithms. Although the results are highly accurate, false positive or negative results can occur. If you still have concerns after consulting our app, please contact your doctor or find a hospital using our locator tool.";
    display:block;
    position:relative;
}
</style>
"""

st.markdown(footer, unsafe_allow_html=True)
