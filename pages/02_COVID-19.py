import streamlit as st
import joblib
import pandas as pd
import sklearn

st.set_page_config(page_title="COVID-19 Predictor", page_icon="")
st.markdown("# COVID-19  Predictor")
st.sidebar.markdown("# COVID-19  Predictor")
st.write("Input your symptoms below")


def covid():
    col1, col2 = st.columns(2)
    breath = col1.checkbox("Do you have problems breathing?")
    fever = col1.checkbox("Do you have a fever?")
    dry_cough = col1.checkbox("Do you have dry cough?")
    sore_throat = col1.checkbox("Do you have a sore throat?")
    hypertension = col1.checkbox("Do you have hypertension?")
    fatigue = col1.checkbox("Do you experience fatigue?")
    travel = col2.checkbox("Have you traveled abroad recently?")
    contact = col2.checkbox(
        "Have you had contact with a COVID patient in the last 14 days?"
    )
    gathering = col2.checkbox(
        "Have you attended a large gathering in the last 14 days?"
    )
    public = col2.checkbox(
        "Have you visited a public, exposed place (e.g. a pool) recently?"
    )
    family = col2.checkbox(
        "Have you had contact with anyone in your family who works in a public, exposed place (e.g. a hospital) recently?"
    )

    # breath = col1.selectbox("Do you have problems breathing?", ["Yes", "No"])
    # fever = col1.selectbox("Do you have a fever?", ["Yes", "No"])
    # dry_cough = col1.selectbox("Do you have dry cough?", ["Yes", "No"])
    # sore_throat = col1.selectbox("Do you have a sore throat?", ["Yes", "No"])
    # hypertension = col1.selectbox("Do you have hypertension?", ["Yes", "No"])
    # fatigue = col1.selectbox("Do you experience fatigue?", ["Yes", "No"])
    # travel = col2.selectbox("Have you traveled abroad recently?", ["Yes", "No"])
    # contact = col2.selectbox(
    #     "Have you had contact with a COVID patient in the last 14 days?", ["Yes", "No"]
    # )
    # gathering = col2.selectbox(
    #     "Have you attended a large gathering in the last 14 days?", ["Yes", "No"]
    # )
    # public = col2.selectbox(
    #     "Have you visited a public exposed place(EG: Pool) recently?", ["Yes", "No"]
    # )
    # family = col2.selectbox(
    #     "Does anyone in your family work in a public exposed place(EG: Hospital) recently?",
    #     ["Yes", "No"],
    # )
    model = joblib.load("covid.pkl")
    input_arr = [
        breath,
        fever,
        dry_cough,
        sore_throat,
        hypertension,
        fatigue,
        travel,
        contact,
        gathering,
        public,
        family,
    ]
    symptoms = [
        "Breathing Problem",
        "Fever",
        "Dry Cough",
        "Sore throat",
        "Hyper Tension",
        "Fatigue ",
        "Abroad travel",
        "Contact with COVID Patient",
        "Attended Large Gathering",
        "Visited Public Exposed Places",
        "Family working in Public Exposed Places",
    ]

    for i in range(len(input_arr)):
        if input_arr[i]:
            input_arr[i] = 1
        else:
            input_arr[i] = 0
    inp = pd.DataFrame([input_arr], columns=symptoms)
    y_pred = model.predict(inp)

    if st.button("Predict"):
        if y_pred[0] == 1:
            st.write(
                """<div style="text-align: center;"><span style="font-size: x-large; color: #ff0000;"><strong>You have a HIGH chance of having and/or carrying COVID-19. Please see a doctor immediately.</strong></span></div>
                <div style="text-align: center;">&nbsp;</div>
                <p style="text-align: center;"><strong>&nbsp;</strong><strong style="background-color: #cc99ff;">There have been approximately 553,850,467 cases of COVID-19 worldwide and 6,363,448 total deaths.</strong></p>
                <p style="text-align: center;">&nbsp;</p>
                <p><em>With an incubation period of 14 days, it is important that early diagnosis is conducted, so that individuals can isolate themselves to reduce the chances that they will infect others and allow themselves to seek treatment earlier, likely reducing disease severity, the risk of long-term disability, and death </em></p>
                <h1 style="text-align: center;">&nbsp;</h1>""",
                unsafe_allow_html=True,
            )
        elif y_pred[0] == 0:
            st.write(
                """<div style="text-align: center;">
                <div><span style="font-size: x-large; color: #ccffcc;"><span style="color: #339966;">You most likely DO NOT have COVID-19. If you are feeling similar symptoms, please get tested.</span></span></div>
                <h1>&nbsp;</h1>
                <p>&nbsp;</p>
                <p><strong>If concerned, there are several ways you can reduce your risk of developing COVID-19, such as:</strong></p>
                <p style="text-align: left;"><span> 1. Giving up and/or avoiding smoking and tobacco.</span></p>
                <p style="text-align: left;"><span> 2. Clean your hands often. Use soap and water, or an alcohol-based hand rub. </span></p>
                <p style="text-align: left;"><span>3. Get vaccinated. Follow local guidance about vaccination.</span></p>
                <p style="text-align: left;"><span> 4. Cover your nose and mouth with your bent elbow or tissue when you cough/sneeze.</span></p>
                <p style="text-align: left;"><span> 5. Stay home if you feel unwell. Open a window if indoors.</span></p>
                <p style="text-align: left;"><span> 6. Wear a mask in public, especially indoors or when physical distancing is not possible.</span></p>
                <p style="text-align: left;"><span>7. Maintain a safe distance from others (at least 1 meter), even if they don&rsquo;t appear to be sick.</span></p>
                <p>&nbsp;</p>
                <p><span style="background-color: #cc99ff;"><strong>There have been approximately 553,850,467 cases of COVID-19 worldwide and 6,363,448 total deaths.</strong></span></p>
                <p>&nbsp;</p>
                <p><em>Luckily there are numerous things we can do to help reduce the spread and avoid serious illness and death. Effective actions like getting the COVID-19 vaccine and a booster shot, wearing a mask, avoiding crowds and indoor places and washing hands often all contribute to fighting the virus.</em></p>
                </div>""",
                unsafe_allow_html=True,
            )


covid()
