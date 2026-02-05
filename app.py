import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("student_pass_model.pkl", "rb"))

st.set_page_config(page_title="AI Student Predictor", layout="centered")

st.markdown("<h1 style='text-align: center;'>🎓 AI Student Predictor</h1>", unsafe_allow_html=True)

st.write("Enter marks to predict whether a student will pass.")

# ✅ SLIDERS (OUTSIDE ANY BRACKET)
reading = st.slider("Reading Score", 0, 100, 50)
writing = st.slider("Writing Score", 0, 100, 50)

# Predict button
if st.button("Predict"):

    input_data = pd.DataFrame(
        [[reading, writing]],
        columns=["reading score", "writing score"]
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("⚠️ Student may FAIL")