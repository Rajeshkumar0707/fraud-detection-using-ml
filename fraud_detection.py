import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('fraud_detection_model.pkl')

st.title("Fraud Detection Model")

st.markdown("Please enter the following details to predict whether a transaction is fraudulent or not:")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance of (SenderOrigin Account)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance of (SenderOrigin Account)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance of (Reciever)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance of (Reciever)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction Result: '{int(prediction)}'")

    if prediction== 1:
        st.error("The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("The transaction is predicted to be LEGITIMATE.")


print("Streamlit app is running. Access it at http://localhost:8501")
