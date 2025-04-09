import streamlit as st
import pandas as pd
from utils.model import load_model
import joblib

st.set_page_config(page_title="UCLA Admission Predictor")

st.title("UCLA Admission Predictor")

st.write("Fill in your academic information to predict admission outcome.")

gre = st.slider("GRE Score", 260, 340, 300)
toefl = st.slider("TOEFL Score", 0, 120, 100)
univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.slider("Statement of Purpose Strength", 1.0, 5.0, 3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)
cgpa = st.slider("CGPA", 6.0, 10.0, 8.5)
research = st.radio("Research Experience", ["No", "Yes"])
research = 1 if research == "Yes" else 0

input_data = pd.DataFrame([[gre, toefl, univ_rating, sop, lor, cgpa, research]],
                          columns=["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"])

scaler = joblib.load("models/scaler.pkl")
model = load_model("models/model.pkl")
X_scaled = scaler.transform(input_data)
prediction = model.predict(X_scaled)[0]

if st.button("Predict"):
    result = "Admitted" if prediction == 1 else "Not Admitted"
    st.subheader("Prediction Result")
    st.success(result)