import streamlit as st
import pandas as pd
import os
import joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_pipeline_final.pkl")
model = joblib.load(MODEL_PATH)
print("CURRENT WORKING DIRECTORY:", os.getcwd())
print("FILES IN CURRENT DIR:", os.listdir())
xgb_pipeline = joblib.load(MODEL_PATH)  # Make sure filename matches
st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("Loan Default Risk Predictor")
st.markdown("Enter borrower details below to calculate default risk:")
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Income = st.number_input("Monthly Income", min_value=0.0, value=50000.0)
LoanAmount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
MonthsEmployed = st.number_input("Months Employed", min_value=0, value=36)
InterestRate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
LoanTerm = st.number_input("Loan Term (months)", min_value=1, value=60)
HasCoSigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])
if st.button("Predict Default Risk"):

    # Create DataFrame EXACTLY like training data
    input_df = pd.DataFrame({
        "Age": [float(Age)],
        "Income": [float(Income)],
        "LoanAmount": [float(LoanAmount)],
        "CreditScore": [float(CreditScore)],
        "MonthsEmployed": [float(MonthsEmployed)],
        "InterestRate": [float(InterestRate)],
        "LoanTerm": [float(LoanTerm)],
        "HasCoSigner": [HasCoSigner]  # DO NOT encode manually
    })

    # Prediction
    prediction = xgb_pipeline.predict(input_df)[0]
    prob = xgb_pipeline.predict_proba(input_df)[0][1]

    # Risk Category Display
    st.subheader("Prediction Result")

    if prob < 0.3:
        st.success("Low Risk")
    elif prob < 0.6:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

    # Probability Display
    st.subheader("Default Probability")
    st.progress(int(prob * 100))
    st.write(f"Probability of Default: **{prob:.2%}**")
    reasons = []

    if CreditScore < 600:
        reasons.append("Low Credit Score")
    if LoanAmount > Income * 5:
        reasons.append("High Loan Amount relative to Income")
    if MonthsEmployed < 12:
        reasons.append("Short Employment History")
    if InterestRate > 15:
        reasons.append("High Interest Rate")
    if HasCoSigner == "No":
        reasons.append("No Co-Signer")

    if reasons:
        st.subheader("Top Risk Drivers")
        for r in reasons[:3]:
            st.markdown(f" {r}")
    else:
        st.write("Borrower appears financially stable.")

