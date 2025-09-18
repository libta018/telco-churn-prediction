import streamlit as st
import pandas as pd
import joblib

# Load saved pipeline
pipe = joblib.load('models/churn_pipeline_v1.joblib')

st.title("📞 Telco Customer Churn Predictor")
st.write("Enter customer details below to predict churn probability:")

# --- Key User Inputs ---
tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
monthly = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=50.0)
contract = st.selectbox('Contract', ['Month-to-month','One year','Two year'])
internet = st.selectbox('Internet Service', ['DSL','Fiber optic','No'])
senior = st.selectbox('Senior Citizen', [0,1])
phoneservice = st.selectbox('Phone Service', ['Yes','No'])

# --- Fill remaining columns with default values ---
default_values = {
    'TotalCharges': tenure * monthly,  # approximate
    'Partner': 'No',
    'Dependents': 'No',
    'PaperlessBilling': 'Yes',
    'gender': 'Female',
    'MultipleLines': 'No',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'PaymentMethod': 'Electronic check',
    'tenure_group': '0-12'
}

# --- Prepare DataFrame for prediction ---
input_df = pd.DataFrame([{
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'Contract': contract,
    'InternetService': internet,
    'SeniorCitizen': senior,
    'PhoneService': phoneservice,
    **default_values
}])

# --- Prediction ---
if st.button("Predict Churn"):
    prob = pipe.predict_proba(input_df)[0,1]
    st.write(f"🔹 Churn Probability: {prob:.2f}")

    if prob > 0.5:
        st.warning("⚠️ High chance of churn!")
    else:
        st.success("✅ Low chance of churn!")
