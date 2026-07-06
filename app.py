import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("churn_model.pkl")

# Title
st.title("📊 Customer Churn Prediction System")

st.write("Enter customer details below")

# -------------------------
# Customer Details
# -------------------------

gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Churn"):

    # Feature Engineering
    average_monthly_spend = total / (tenure + 1)

    is_new_customer = 1 if tenure < 12 else 0

    high_monthly_charge = 1 if monthly > 70 else 0

    sample = pd.DataFrame({

        "gender":[gender],

        "SeniorCitizen":[senior],

        "Partner":[partner],

        "Dependents":[dependents],

        "tenure":[tenure],

        "PhoneService":[phone_service],

        "MultipleLines":[multiple_lines],

        "InternetService":[internet_service],

        "OnlineSecurity":[online_security],

        "OnlineBackup":[online_backup],

        "DeviceProtection":[device_protection],

        "TechSupport":[tech_support],

        "StreamingTV":[streaming_tv],

        "StreamingMovies":[streaming_movies],

        "Contract":[contract],

        "PaperlessBilling":[paperless],

        "PaymentMethod":[payment],

        "MonthlyCharges":[monthly],

        "TotalCharges":[total],

        "AverageMonthlySpend":[average_monthly_spend],

        "IsNewCustomer":[is_new_customer],

        "HighMonthlyCharge":[high_monthly_charge]

    })

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )