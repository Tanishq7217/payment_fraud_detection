from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('fraud_model.pkl')

app =FastAPI(title = "UPI Fraud Detection API")

class Transaction(BaseModel):
     transaction_amount: float
     transaction_hour: int
     account_age_days: int
     customer_age: int
     quantity: int
     payment_method: int
     product_category: int
     device_used: int
     address_mismatch: int

@app.get("/")
def home():
     return {"message" : "UPI Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to dataframe
    data = pd.DataFrame([{
        'Transaction Amount': transaction.transaction_amount,
        'Transaction Hour': transaction.transaction_hour,
        'Account Age Days': transaction.account_age_days,
        'Customer Age': transaction.customer_age,
        'Quantity': transaction.quantity,
        'Payment Method': transaction.payment_method,
        'Product Category': transaction.product_category,
        'Device Used': transaction.device_used,
        'Address Mismatch': transaction.address_mismatch
    }])

    # Get prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "transaction_amount": transaction.transaction_amount,
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability) * 100, 2),
        "risk_level": "HIGH RISK" if probability > 0.7 else "MEDIUM RISK" if probability > 0.3 else "LOW RISK",
        "action": "BLOCK TRANSACTION" if prediction == 1 else "ALLOW TRANSACTION"
    }
