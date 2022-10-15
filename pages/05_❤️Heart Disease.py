import streamlit as st
import joblib
import pandas as pd
import pickle
import math

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.markdown("# Heart Disease  Predictor")
st.write("This predicts the chance of having Heart Disease through Machine Learning. Designed for doctors in assisting the diagnosis of Heart Disase, this algorithm requires unconventional measuresments that may not be readily available patient.")
st.write("Input your symptoms below")

col1, col2 = st.columns(2)
age = col1.number_input("How old are you?", value=29, min_value=29, max_value=77)  # FIX
theSet = [
    29,
    34,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    74,
    76,
    77,
]
if theSet.count(age) == 0:
    minVal = 9999999
    newVal = 0
    for i in theSet:
        x = abs(age - i)
        if x < minVal:
            minVal = x
            newVal = i
    age = newVal


trestbps = col1.number_input(
    "What is your resting systolic blood pressure (mmHg)?",
    value=120,
    min_value=94,
    max_value=200,
)
theSet = [
    94,
    100,
    101,
    102,
    104,
    105,
    106,
    108,
    110,
    112,
    114,
    115,
    117,
    118,
    120,
    122,
    123,
    124,
    125,
    126,
    128,
    129,
    130,
    132,
    134,
    135,
    136,
    138,
    140,
    142,
    144,
    145,
    146,
    148,
    150,
    152,
    154,
    155,
    156,
    160,
    164,
    165,
    170,
    172,
    174,
    178,
    180,
    192,
    200,
]
if theSet.count(trestbps) == 0:
    minVal = 9999999
    newVal = 0
    for i in theSet:
        x = abs(trestbps - i)
        if x < minVal:
            minVal = x
            newVal = i
    trestbps = newVal


chol = col1.number_input(
    "What is your cholesterol level (mg/dL)?", value=250, min_value=126, max_value=417
)  # FIX
theSet = [
    126,
    131,
    141,
    149,
    157,
    160,
    164,
    166,
    167,
    168,
    169,
    172,
    174,
    175,
    176,
    177,
    178,
    180,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    192,
    193,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    273,
    274,
    275,
    276,
    277,
    278,
    281,
    282,
    283,
    284,
    286,
    288,
    289,
    290,
    293,
    294,
    295,
    298,
    299,
    300,
    302,
    303,
    304,
    305,
    306,
    307,
    308,
    309,
    311,
    313,
    315,
    318,
    319,
    321,
    322,
    325,
    326,
    327,
    330,
    335,
    340,
    341,
    342,
    353,
    354,
    360,
    394,
    407,
    409,
    417,
    564,
]
if theSet.count(chol) == 0:
    minVal = 9999999
    newVal = 0
    for i in theSet:
        x = abs(chol - i)
        if x < minVal:
            minVal = x
            newVal = i
    chol = newVal

thalach = col1.number_input(
    "What was your maximum heart rate achieved?", value=150, min_value=71, max_value=202
)
theSet = [
    71,
    88,
    90,
    95,
    96,
    97,
    99,
    103,
    105,
    106,
    108,
    109,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    177,
    178,
    179,
    180,
    181,
    182,
    184,
    185,
    186,
    187,
    188,
    190,
    192,
    194,
    195,
    202,
]
if theSet.count(thalach) == 0:
    minVal = 9999999
    newVal = 0
    for i in theSet:
        x = abs(thalach - i)
        if x < minVal:
            minVal = x
            newVal = i
    thalach = newVal

oldpeak = col2.number_input(
    "How much of an ST depression induced by exercise relative to rest do you have (ECG)?",
    value=0.0,
    min_value=0.0,
    step=0.1,
    max_value=6.2,
)
theSet = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.8,
    1.9,
    2.0,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5,
    2.6,
    2.8,
    2.9,
    3.0,
    3.1,
    3.2,
    3.4,
    3.5,
    3.6,
    3.8,
    4.0,
    4.2,
    4.4,
    5.6,
    6.2,
]
if theSet.count(oldpeak) == 0:
    minVal = 9999999
    newVal = 0
    for i in theSet:
        x = abs(oldpeak - i)
        if x < minVal:
            minVal = x
            newVal = i
    oldpeak = newVal

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
    index=1,
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
        st.write(
                """<div style="text-align: center;">
                <div><span style="font-size: x-large; background-color: #ff6600;">You have a HIGH chance of having a heart and cardiovascular disease. Please see a doctor immediately.</span></div>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p><strong>Cardiovascular diseases (CVDs) are the leading cause of death globally. An estimated 17.9 million people die from cardiovascular diseases yearly, representing 32% of all global deaths.</strong></p>
                <p>&nbsp;</p>
                <p><em>Identifying those at the highest risk of CVDs, diagnosing as early as possible, and ensuring patients receive appropriate treatment at the correct time can prevent premature and consequential deaths. Access to noncommunicable disease medicines and basic health technologies is essential to ensure that those in need receive appropriate care.</em></p>
                <h1>&nbsp;</h1>
                <h1>&nbsp;</h1>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <h1>&nbsp;</h1>
                <h1>&nbsp;</h1>
                <p>&nbsp;</p>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.write(
                """<div style="text-align: center;"><span style="font-size: x-large; background-color: #00ff00;">You most likely DO NOT have heart and/or cardiovascular disease.</span></div>
                <p><strong>If concerned, there are several ways you can reduce your risk of developing heart and/or cardiovascular diseases, such as:</strong></p>
                <p><span>1. Lowering your blood and cholesterol levels.</span></p>
                <p><span> 2. Eating a healthy, balanced diet.</span></p>
                <p><span> 3. Maintaining a healthy weight.</span></p>
                <p><span> 4. Giving up and/or avoiding smoking and tobacco.</span></p>
                <p><span> 5. Reducing alcohol consumption.</span></p>
                <p><span> 6. Keeping blood pressure under control.</span></p>
                <p><span> 7. Being consistently active and involved in physical activity.</span></p>
                <p>&nbsp;</p>
                <p><strong>Cardiovascular diseases (CVDs) are the leading cause of death globally. An estimated 17.9 million people died from cardiovascular diseases per year, representing 32% of all global deaths.</strong></p>
                <p>&nbsp;</p>
                <p><em>Identifying those at the highest risk of CVDs early on, diagnosing as early as possible, and ensuring patients receive appropriate treatment at the correct time can prevent premature and consequential deaths. Access to noncommunicable disease medicines and basic health technologies is essential to ensure that those in need receive appropriate care.</em></p>
                <h1>&nbsp;</h1>
                <h1>&nbsp;</h1>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>
                <p>&nbsp;</p>""",
                unsafe_allow_html=True,
            )
