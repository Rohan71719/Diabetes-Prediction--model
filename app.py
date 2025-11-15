# ================================================================
# 💉 Diabetes Prediction Web App using Streamlit
# ================================================================

import streamlit as st
import numpy as np
import joblib

# -------------------------------
# 🧠 Load trained model and scaler
# -------------------------------
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------------------
# 🎨 Page title and description
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction App", page_icon="💉")
st.title("💉 Diabetes Prediction App")
st.markdown("Enter your health details below to check your diabetes risk.")

# -------------------------------
# 🧾 Input Fields (User Form)
# -------------------------------
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=140, value=70)
skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age (years)", min_value=10, max_value=100, value=30)

# -------------------------------
# 🎯 Prediction Button
# -------------------------------
if st.button("🔍 Predict"):
    # Prepare features for prediction
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]  # probability of diabetes

    # -------------------------------
    # 🟢🟡🔴 Display Result
    # -------------------------------
    st.subheader("Prediction Result:")

    if prediction[0] == 1:
        st.error(f"🔴 High Risk of Diabetes! (Probability: {probability:.2f})")
    else:
        st.success(f"🟢 Low Risk of Diabetes (Probability: {probability:.2f})")

    # Show progress bar visualization
    st.progress(int(probability * 100))
