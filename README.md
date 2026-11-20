# E-Commerce Payment Fraud Detection

An XGBoost model that scores e-commerce payments for fraud risk, served through a FastAPI endpoint and analysed in a two-page Power BI dashboard.

Trained on ~1.5 million transactions from the [Fraudulent E-Commerce Transactions dataset](https://www.kaggle.com/) (Kaggle), with payments made by credit card, debit card, PayPal and bank transfer.

## What it does
- Scores each transaction with a fraud probability (0–100%)
- Assigns a risk level: LOW (< 30%), MEDIUM (30–70%), HIGH (> 70%)
- Exposes the model as a REST API for real-time scoring
- Visualises fraud patterns and model performance in Power BI

## Model Performance
Evaluated on a held-out test set of **299,318 transactions** that the model never saw during training (80/20 split).

| Metric | Value |
|---|---|
| Recall (fraud caught) | **70.1%** |
| Precision (flags that are real fraud) | **15.0%** |
| F1 score | 0.25 |
| Transactions flagged | 23.5% (actual fraud rate: 5.1%) |

**Trade-off:** the model uses `scale_pos_weight=19` to handle the 95/5 class imbalance, which favours catching fraud over avoiding false alarms. At the default 0.5 threshold it raises about 5.6 false alarms per real fraud caught. The Power BI dashboard includes a threshold slider that shows how precision and recall shift as the cut-off changes.

## Key Findings
- **Order amount** is the strongest signal: orders above 1,000 have a 56% fraud rate, against 3–5% for typical orders
- **New accounts** (under 30 days old) have a 22.6% fraud rate, about 7× higher than older accounts (3.1%)
- **Time of day:** fraud runs at 10.7% between 12 AM and 5 AM, compared with about 3% for the rest of the day
- **Payment method, device and billing/shipping address mismatch** show no meaningful difference (each around 5%)

## Tech Stack
- **ML:** Python, Pandas, Scikit-learn, XGBoost, Joblib
- **API:** FastAPI, Uvicorn
- **BI:** Power BI (DAX, what-if parameter)

## Project Structure
```
├── upi_fraud_detection.py   # Exploratory data analysis
├── model.py                 # Feature engineering + XGBoost training
├── export_predictions.py    # Scores the test set and exports it for Power BI
├── api.py                   # FastAPI prediction endpoint
└── README.md
```

## How to Run
1. Download both dataset CSVs from Kaggle into the project folder.
2. Install the dependencies:
   ```
   pip install pandas scikit-learn xgboost joblib fastapi uvicorn
   ```
3. Train the model, which writes `fraud_model.pkl`:
   ```
   python model.py
   ```
4. Export the scored test set for Power BI, which writes `fraud_predictions.csv`:
   ```
   python export_predictions.py
   ```
5. Start the API:
   ```
   uvicorn api:app --reload
   ```

## API Usage
`POST /predict`
```json
{
  "transaction_amount": 1500.00,
  "transaction_hour": 2,
  "account_age_days": 5,
  "customer_age": 22,
  "quantity": 1,
  "payment_method": 1,
  "product_category": 1,
  "device_used": 0,
  "address_mismatch": 1
}
```

Response:
```json
{
  "transaction_amount": 1500.0,
  "is_fraud": true,
  "fraud_probability": 99.53,
  "risk_level": "HIGH RISK",
  "action": "BLOCK TRANSACTION"
}
```

**Category encoding** (alphabetical label codes, as used in training):
- `payment_method`: 0 = PayPal, 1 = bank transfer, 2 = credit card, 3 = debit card
- `device_used`: 0 = desktop, 1 = mobile, 2 = tablet
- `product_category`: product categories in alphabetical order, starting at 0

## Power BI Dashboard
**Page 1: Fraud Overview**
- KPIs: total transactions, fraud count, fraud rate, fraud amount
- Fraud rate by transaction hour, account age, order amount, payment method, device and address status
- Date, payment method, device and risk level slicers

**Page 2: Model Performance**
- Interactive threshold slider with live precision and recall
- Confusion matrix
- Actual vs. predicted fraud by hour
- Fraud rate by risk level and probability band

**Page 3: Investigation**
- Transaction-level table of high-risk payments, sortable by fraud probability and colour-coded by outcome

## Limitations and Next Steps
- The dataset is synthetic. Patterns may not carry over to real payment data.
- Low precision means a 0.5 threshold would block too many genuine customers in production. A better policy would be to block only HIGH risk and send MEDIUM risk to manual review.
- The API takes categorical fields as numeric codes. It should accept text labels and encode them on the server.
- Next steps: threshold tuning, cost-based evaluation, SHAP explanations for each prediction.
