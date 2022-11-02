import sklearn as sk
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import pandas as pd
import pickle

st.set_page_config(page_title="General Disease Predictor", page_icon="🦠")
filename = "gendisease/gendisease.sav"

model = pickle.load(open(filename, "rb"))

symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "spotting_urination",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss",
    "restlessness",
    "lethargy",
    "patches_in_throat",
    "irregular_sugar_level",
    "cough",
    "high_fever",
    "sunken_eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "pain_behind_the_eyes",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of_urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhos)",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic_patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "fluid_overload.1",
    "history_of_alcohol_consumption",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]
Data = "gendisease/disease_input.csv"
data = pd.read_csv(Data).dropna(axis=1)
encoder = LabelEncoder()
# dropping all the uncessary prognosis - ones that are quite obvious
data = data[data.prognosis != "Drug Reaction"]
data = data[data.prognosis != "Acne"]
data = data[data.prognosis != "AIDS"]
data = data[data.prognosis != "Paralysis (brain hemorrhage)"]
data = data[data.prognosis != "Heart attack"]

# st.write(data)
data["prognosis"] = encoder.fit_transform(data["prognosis"])
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {"symptom_index": symptom_index, "predictions_classes": encoder.classes_}


def predictDisease(symptoms):
    if symptoms == "":
        return "No disease"
    else:
        symptoms = symptoms.split(",")

        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1

        input_data = np.array(input_data).reshape(1, -1)
        svm_prediction = data_dict["predictions_classes"][model.predict(input_data)[0]]
        return svm_prediction


st.header("General Disease Predictor")
st.write(
    "Choose any symptoms that you are experiencing based on the category, and we will help to diagnose the disease."
)

skin = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "cold_hands_and_feets",
    "yellowish_skin",
    "puffy_face_and_eyes",
    "brittle_nails",
    "swollen_extremeties",
    "blister",
    "inflammatory_nails",
    "small_dents_in_nails",
    "skin_peeling",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "red_spots_over_body",
    "dischromic_patches",
]

digestion = [
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "vomiting",
    "weight_gain",
    "weight_loss",
    "patches_in_throat",
    "irregular_sugar_level",
    "indigestion",
    "loss_of_appetite",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "swelling_of_stomach",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "stomach_bleeding",
    "irritation_in_anus",
    "cramps",
    "excessive_hunger",
    "passage_of_gases",
    "belly_pain",
    "increased_appetite",
    "distention_of_abdomen",
    "nausea",
    "acute_liver_failure",
]

ent_respiratory = [
    "continuous_sneezing",
    "cough",
    "breathlessness",
    "runny_nose",
    "congestion",
    "sinus_pressure",
    "patches_in_throat",
    "phlegm",
    "throat_irritation",
    "loss_of_smell",
    "mucoid_sputum",
    "rusty_sputum",
    "blood_in_sputum",
]

eyes = [
    "sunken_eyes",
    "pain_behind_the_eyes",
    "yellowing_of_eyes",
    "blurred_and_distorted_vision",
    "redness_of_eyes",
    "watering_from_eyes",
    "visual_disturbances",
]

musculoskeletal = [
    "joint_pain",
    "muscle_wasting",
    "weakness_in_limbs",
    "cramps",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "muscle_pain",
    "painful_walking",
]

body_temperature = ["shivering", "chills", "high_fever", "sweating", "mild_fever"]

urinary = [
    "burning_micturition",
    "spotting_urination",
    "dark_urine",
    "yellow_urine",
    "bladder_discomfort",
    "foul_smell_of_urine",
    "continuous_feel_of_urine",
    "polyuria",
]

emotional = [
    "anxiety",
    "lethargy",
    "depression",
    "irritability",
    "fatigue",
    "restlessness",
    "mood_swings",
]

circulatory = [
    "irregular_sugar_level",
    "fast_heart_rate",
    "swollen_blood_vessels",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "prominent_veins_on_calf",
    "palpitations",
]

other = []
for i in symptoms:
    if (
        i in skin
        or i in digestion
        or i in ent_respiratory
        or i in eyes
        or i in musculoskeletal
        or i in body_temperature
        or i in urinary
        or i in emotional
        or i in circulatory
    ):
        pass
    else:
        other.append(i)
for i in range(len(other)):
    other[i] = other[i].replace("_", " ")
    other[i] = other[i].title()
for i in range(len(skin)):
    skin[i] = skin[i].replace("_", " ")
    skin[i] = skin[i].title()
# st.subheader("Skin")
co1, col2 = st.columns(2)

skinselect = co1.multiselect(
    "Choose any symptoms that are related to the integumentary system", skin
)
for i in range(len(digestion)):
    digestion[i] = digestion[i].replace("_", " ")
    digestion[i] = digestion[i].title()
# st.subheader("Digestive System")
digestionselect = co1.multiselect(
    "Choose any symptoms that are related to the digestive system", digestion
)
for i in range(len(ent_respiratory)):
    ent_respiratory[i] = ent_respiratory[i].replace("_", " ")
    ent_respiratory[i] = ent_respiratory[i].title()
# st.subheader("ENT/Respiratory System")
respiratoryselect = co1.multiselect(
    "Choose any symptoms that are related to the ENT/respiratory system",
    ent_respiratory,
)
for i in range(len(eyes)):
    eyes[i] = eyes[i].replace("_", " ")
    eyes[i] = eyes[i].title()
# st.subheader("Eyes")
eyeselect = co1.multiselect(
    "Choose any symptoms that are related to your eyes",
    eyes,
)
for i in range(len(musculoskeletal)):
    musculoskeletal[i] = musculoskeletal[i].replace("_", " ")
    musculoskeletal[i] = musculoskeletal[i].title()
# st.subheader("Musculoskeletal System")
musculoskeletalselect = co1.multiselect(
    "Choose any symptoms that are related to your musculoskeletal system",
    musculoskeletal,
)
for i in range(len(body_temperature)):
    body_temperature[i] = body_temperature[i].replace("_", " ")
    body_temperature[i] = body_temperature[i].title()
# st.subheader("Body Temperature")
temperatureselect = col2.multiselect(
    "Choose any symptoms that are related to your body temperature",
    body_temperature,
)
for i in range(len(urinary)):
    urinary[i] = urinary[i].replace("_", " ")
    urinary[i] = urinary[i].title()
# st.subheader("Urinary System")
urinaryselect = col2.multiselect(
    "Choose any symptoms that are related to your urinary system",
    urinary,
)
for i in range(len(emotional)):
    emotional[i] = emotional[i].replace("_", " ")
    emotional[i] = emotional[i].title()
# st.subheader("Emotional/Energy")
emotionselect = col2.multiselect(
    "Choose any symptoms that are related to your emotions/level of energy",
    emotional,
)
for i in range(len(circulatory)):
    circulatory[i] = circulatory[i].replace("_", " ")
    circulatory[i] = circulatory[i].title()
# st.subheader("Circulatory System")
circulatoryselect = col2.multiselect(
    "Choose any symptoms that are related to your circulatory system",
    circulatory,
)
# st.subheader("Other")
otherselect = col2.multiselect("Choose any other symptoms", other)
patientsymp = (
    skinselect
    + digestionselect
    + respiratoryselect
    + eyeselect
    + musculoskeletalselect
    + temperatureselect
    + urinaryselect
    + emotionselect
    + circulatoryselect
    + otherselect
)
finalstr = ""
for i in range(len(patientsymp)):
    if i != len(patientsymp) - 1:
        finalstr += patientsymp[i] + ","
    else:
        finalstr += patientsymp[i]
if st.button("Predict"):
    st.write(
        """<h1 style="text-align:center">Your condition is most likely:</h1>

		""",
        unsafe_allow_html=True,
    )
    st.write(
        """<h1 style="text-align:center"><span style="font-family:
	Comic Sans MS,cursive"><span style="background-color:#e74c3c">"""
        + str(predictDisease(finalstr))
        + """</span></span></h1>
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
