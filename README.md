# UPI Fraud Detection System 

ML-powered fraud detection system trained on 1.5 million transactions.

## What it does
- Detects fraudulent UPI transactions in real time
- Returns fraud probability score and risk level
- Blocks high risk transactions automatically

## Tech Stack
- Python, XGBoost, Scikit-learn, Pandas
- FastAPI, Uvicorn
- Power BI Dashboard
- Joblib for model serialization

## Model Performance
- Trained on 1.5 million transactions
- 70% fraud detection rate (recall)
- Deployed as REST API

## Project Structure
upi-fraud-detection/

├── upi_fraud_detection.py  # Data exploration

├── model.py                # ML model training

├── api.py                  # FastAPI deployment

├── export_predictions.py   # Power BI data export

└── README.md

## API Usage
Start the API:

uvicorn api:app --reload

Send a transaction:
```json
POST /predict
{
  "transaction_amount": 1500.00,
  "transaction_hour": 2,
  "account_age_days": 5,
  "customer_age": 22,
  "quantity": 1,
  "payment_method": 0,
  "product_category": 1,
  "device_used": 0,
  "address_mismatch": 1
}
```

Response:
```json
{
  "is_fraud": true,
  "fraud_probability": 99.53,
  "risk_level": "HIGH RISK",
  "action": "BLOCK TRANSACTION"
}
```

## Dashboard
Power BI dashboard showing:
- Total transactions and fraud detected
- Fraud by hour, location, payment method
- Risk level breakdown
