import subprocess
import threading

# Streamlit App Code as a String
streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import zipfile

st.set_page_config(page_title="Real-Time Loan Risk Predictor", page_icon="🔄", layout="centered")

st.title("🔄 Real-Time Loan Risk Predictor")
st.markdown("Quickly assess the likelihood of loan default based on key financial features.")

# Load the dataset
DATA_FILE = "trainnn.csv"
ZIP_FILE = "trainnn.csv.zip"

if not os.path.exists(DATA_FILE) and os.path.exists(ZIP_FILE):
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
        
df = pd.read_csv(DATA_FILE)

# Split features/target
X = df.drop(columns="Default")
y = df["Default"]

# Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Transformers
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Model setup
model = RandomForestClassifier(random_state=42)
pipe = Pipeline([
    ("preprocess", preprocessor),
    ("clf", model)
])

# Train the model
pipe.fit(X, y)

# Real-time prediction form
st.write("### 📋 Enter Applicant Details")

user_data = {}
user_data["Age"] = st.slider("Age", 18, 100, 30)
user_data["Income"] = st.number_input("Annual Income", min_value=0, value=50000)
user_data["LoanAmount"] = st.number_input("Loan Amount", min_value=0, value=10000)
user_data["CreditScore"] = st.slider("Credit Score", 300, 850, 650)
user_data["MonthsEmployed"] = st.number_input("Months Employed", min_value=0, value=12)
user_data["NumCreditLines"] = st.slider("Number of Credit Lines", 0, 20, 4)
user_data["InterestRate"] = st.slider("Interest Rate (%)", 0.0, 50.0, 12.0)
user_data["LoanTerm"] = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
user_data["DTIRatio"] = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
user_data["Education"] = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
user_data["EmploymentType"] = st.selectbox("Employment Type", ["Full-time", "Part-time", "Unemployed", "Self-employed"])
user_data["MaritalStatus"] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
user_data["HasMortgage"] = st.selectbox("Has Mortgage?", ["Yes", "No"])
user_data["HasDependents"] = st.selectbox("Has Dependents?", ["Yes", "No"])
user_data["LoanPurpose"] = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
user_data["HasCoSigner"] = st.selectbox("Has Co-signer?", ["Yes", "No"])

# Create a DataFrame from the user input
user_input_df = pd.DataFrame([user_data])

# Make a prediction
if st.button("Predict Default Risk"):
    # Transform user input
    user_input_transformed = preprocessor.transform(user_input_df)
    
    # Predict the risk
    prediction = pipe.predict(user_input_transformed)[0]
    proba = pipe.predict_proba(user_input_transformed)[0][1]

    if prediction == 1:
        st.error(f"❌ Prediction: Likely to Default (Risk Score: {proba:.2f})")
    else:
        st.success(f"✅ Prediction: Likely to Repay (Risk Score: {proba:.2f})")

st.markdown("---")
st.markdown("© 2025 **Real-Time Loan Risk Predictor**")
"""

# Save the Streamlit app to a file
with open("realtime_predictor.py", "w") as f:
    f.write(streamlit_code)

# Run Streamlit in a separate thread
def run_streamlit():
    subprocess.run(["streamlit", "run", "realtime_predictor.py", "--server.port", "8501"])

threading.Thread(target=run_streamlit).start()

# Start Cloudflared tunnel
!./cloudflared tunnel --url http://localhost:8501
