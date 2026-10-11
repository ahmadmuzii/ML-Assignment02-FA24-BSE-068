"""
Heart Disease Prediction System - Streamlit web app (Task 2, FA24-BSE-068)
-------------------------------------------------------------------------
Lets anyone use the Support Vector Classifier trained in
'Heart_Disease_Prediction.ipynb' from a browser.

1. Loads 'heart_disease_svc_trained_model.pkl' saved in Step 6.4.
2. Fits the same LabelEncoders as Step 5.1 on the same value lists, so each
   text value gets exactly the code the model saw during training.
3. Encodes the 4 inputs chosen in the form and shows the prediction.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_trained_model.pkl"

# Possible values of each input attribute (same lists as Step 5.1 of the notebook).
# The dict order is also the column order the model was trained on.
ATTRIBUTE_VALUES = {
    "ChestPainType": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    "ExerciseInducedAngina": ["Yes", "No"],
    "MajorVessels": ["Zero", "One", "Two", "Three"],
    "Thalassemia": ["Normal", "Fixed Defect", "Reversible Defect"],
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


@st.cache_resource
def build_label_encoders():
    # LabelEncoder codes depend only on the sorted value list, so these match training
    return {name: LabelEncoder().fit(values) for name, values in ATTRIBUTE_VALUES.items()}


model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Sidebar: model facts
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("About the model")
    st.markdown(
        """
        - **Algorithm:** Support Vector Classifier (SVC)
        - **Dataset:** UCI Heart Disease, Cleveland (297 patients after removing missing values)
        - **Split:** 80% training / 20% testing
        - **Test accuracy:** 85%
        - **Recall (heart disease class):** 82%
        """
    )
    st.markdown("[UCI dataset page](https://archive.ics.uci.edu/dataset/45/heart+disease)")

# ---------------------------------------------------------------------------
# Main page: input form
# ---------------------------------------------------------------------------

st.title("🫀 Heart Disease Prediction System")
st.write(
    "Select the results of four cardiac tests. The trained Support Vector Classifier "
    "predicts whether heart disease is present."
)

with st.form("patient_form"):
    left, right = st.columns(2)
    with left:
        chest_pain_type = st.selectbox(
            "Chest pain type", ATTRIBUTE_VALUES["ChestPainType"],
            help="Type of chest pain the patient reports (UCI attribute: cp).")
        major_vessels = st.selectbox(
            "Major vessels coloured by fluoroscopy", ATTRIBUTE_VALUES["MajorVessels"],
            help="Number of major blood vessels (0-3) seen on fluoroscopy (UCI attribute: ca).")
    with right:
        exercise_induced_angina = st.radio(
            "Chest pain during exercise test?", ["No", "Yes"], horizontal=True,
            help="Exercise-induced angina (UCI attribute: exang).")
        thalassemia = st.selectbox(
            "Thallium scan result", ATTRIBUTE_VALUES["Thalassemia"],
            help="Result of the thallium stress scan (UCI attribute: thal).")
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction (same as Steps 8.2 - 8.5 of the notebook)
# ---------------------------------------------------------------------------

if submitted:
    user_input = pd.DataFrame({
        "ChestPainType": [chest_pain_type],
        "ExerciseInducedAngina": [exercise_induced_angina],
        "MajorVessels": [major_vessels],
        "Thalassemia": [thalassemia],
    })

    unseen_data_features = pd.DataFrame(
        {name: encoders[name].transform(user_input[name]) for name in ATTRIBUTE_VALUES}
    )

    predicted_heart_disease = model.predict(unseen_data_features)[0]

    if predicted_heart_disease == 1:
        st.error("### Prediction: Heart disease present")
        st.write("This combination of test results matches patients who had heart disease in the training data.")
    else:
        st.success("### Prediction: No heart disease")
        st.write("This combination of test results matches patients without heart disease in the training data.")

    with st.expander("Show the input and the encoded feature vector"):
        st.dataframe(user_input, hide_index=True)
        st.dataframe(unseen_data_features, hide_index=True)

st.divider()
st.caption(
    "Educational project for AIC354 Machine Learning Fundamentals, COMSATS University Islamabad, "
    "Lahore Campus (FA24-BSE-068). Not a medical diagnostic tool."
)
