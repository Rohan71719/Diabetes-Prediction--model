🩺 Diabetes Prediction System
A Machine Learning Project by

Rohan Sen • Tanmoy Sarkar • Sohom Chatterjee

📌 Overview

This project is a Diabetes Prediction System built using Machine Learning (Logistic Regression) and deployed through a Streamlit web application.
The system predicts whether a person is likely to have diabetes based on medical input features.

This project is based on the Pima Indian Diabetes Dataset.

🎯 Objective

To build an easy-to-use, fast, and accurate diabetes risk prediction tool using ML techniques, helping in early screening and awareness.


🛠️ Technologies Used

Python

Scikit-learn

Pandas, NumPy

Matplotlib / Seaborn (for analysis)

StandardScaler (Feature Scaling)

Logistic Regression (ML Model)

Streamlit

Joblib (model saving)

├── data/
│   └── diabetes.csv
├── model/
│   └── diabetes_model.pkl
├── app.py
├── preprocess.py
├── requirements.txt
├── README.md
└── assets/
    └── screenshots/

🧪 How It Works

User enters medical values:

Glucose

Blood Pressure

BMI

Insulin

Pregnancies

Age

Skin Thickness

Diabetes Pedigree Function

Data gets scaled using StandardScaler.

ML model predicts the probability of diabetes.

Streamlit app displays:

Result (Diabetic / Non-Diabetic)

Probability bar

Helpful color-coded output
