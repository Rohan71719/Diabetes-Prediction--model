# ================================================================
# 🧠 Diabetes Prediction Model Training Script
# ================================================================
# This script:
# 1. Loads the Pima Indians Diabetes Dataset
# 2. Cleans and preprocesses data
# 3. Trains a Logistic Regression model
# 4. Evaluates performance
# 5. Saves the trained model and scaler for Streamlit use
# ================================================================

# -------------------------------
# 📦 Import Required Libraries
# -------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# -------------------------------
# 📊 Load Dataset
# -------------------------------
# Make sure 'diabetes.csv' is in the same folder as this script.
# If not, provide the full path or use the UCI link.

df = pd.read_csv('diabetes.csv')
print("✅ Dataset loaded successfully!")
print(df.head(), "\n")

# -------------------------------
# 🧹 Handle Missing or Zero Values
# -------------------------------
# These columns shouldn't have 0 values.
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_with_zero:
    df[col] = df[col].replace(0, df[col].mean())

print("✅ Zero/invalid values replaced with column mean.\n")

# -------------------------------
# ✂️ Split Dataset
# -------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📚 Training samples:", X_train.shape[0])
print("🧪 Testing samples:", X_test.shape[0], "\n")

# -------------------------------
# ⚖️ Standardize Features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features standardized.\n")

# -------------------------------
# 🤖 Train Logistic Regression Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -------------------------------
# 📈 Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_scaled)

print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")
print("📋 Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%\n")

# -------------------------------
# 💾 Save Model and Scaler
# -------------------------------
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and Scaler saved successfully!")
print("   → diabetes_model.pkl")
print("   → scaler.pkl")
