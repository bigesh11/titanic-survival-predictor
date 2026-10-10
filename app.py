"""Streamlit app: Titanic survival predictor.

Run locally:  streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import ALL_FEATURES, EMBARKED_MAP

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")


@st.cache_resource
def load_model():
    """Load the saved scikit-learn pipeline (preprocessing + model) once."""
    return joblib.load("models/titanic_model.joblib")


model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter a passenger's details and the model will estimate their chance of surviving the Titanic disaster.")

# ---------- User inputs ----------
col1, col2 = st.columns(2)
with col1:
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    age = st.slider("Age", 0, 80, 30)
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2,
                          format_func=lambda c: {1: "1st (upper)", 2: "2nd (middle)", 3: "3rd (lower)"}[c])
with col2:
    fare = st.number_input("Ticket fare (£)", min_value=0.0, max_value=520.0, value=15.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", list(EMBARKED_MAP.keys()),
                            index=2, format_func=lambda k: EMBARKED_MAP[k])
    sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / children aboard", 0, 9, 0)

# ---------- Build the exact features the model was trained on ----------
family_size = int(sibsp + parch + 1)
row = pd.DataFrame([{
    "Age": age,
    "Fare_log": np.log1p(fare),
    "FamilySize": family_size,
    "Pclass": pclass,
    "Embarked": embarked,
    "Sex": 1 if sex == "Female" else 0,
    "IsAlone": int(family_size == 1),
}])[ALL_FEATURES]

# ---------- Predict ----------
if st.button("Predict survival", type="primary"):
    prob = float(model.predict_proba(row)[0, 1])
    if prob >= 0.5:
        st.success(f"✅ Likely to SURVIVE  (probability {prob:.0%})")
    else:
        st.error(f"❌ Likely NOT to survive  (probability {prob:.0%})")
    st.progress(prob)
    st.caption(f"Family size: {family_size} · Travelling alone: {'yes' if family_size == 1 else 'no'}")

with st.expander("About this model"):
    st.write(
        "Trained on 891 labeled Titanic passengers using a scikit-learn pipeline "
        "(scaling + one-hot encoding + the best model from cross-validated Grid Search). "
        "It is a learning project based on historical data, not for real decisions."
    )
