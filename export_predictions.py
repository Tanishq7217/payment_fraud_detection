import pandas as pd
import joblib

# Load data
print("Loading data...")
df1 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data.csv")
df2 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data_2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Feature engineering - use same column names as training
df['Address Mismatch'] = (df['Billing Address'] != df['Shipping Address']).astype(int)
df['Payment Method'] = df['Payment Method'].astype('category').cat.codes
df['Product Category'] = df['Product Category'].astype('category').cat.codes
df['Device Used'] = df['Device Used'].astype('category').cat.codes

# Load saved model
print("Loading model...")
model = joblib.load('fraud_model.pkl')

# Features - exact same names as training
features = [
    'Transaction Amount',
    'Transaction Hour',
    'Account Age Days',
    'Customer Age',
    'Quantity',
    'Payment Method',
    'Product Category',
    'Device Used',
    'Address Mismatch'
]

# Predict on sample of 50,000 rows (full data is too big for Power BI)
print("Predicting...")
sample = df.sample(50000, random_state=42)
X_sample = sample[features]

sample['Predicted Fraud'] = model.predict(X_sample)
sample['Fraud Probability'] = model.predict_proba(X_sample)[:, 1]
sample['Risk Level'] = sample['Fraud Probability'].apply(
    lambda x: 'HIGH' if x > 0.7 else 'MEDIUM' if x > 0.3 else 'LOW'
)

# Keep useful columns
output = sample[[
    'Transaction Amount',
    'Transaction Date',
    'Payment Method',
    'Product Category',
    'Device Used',
    'Customer Age',
    'Customer Location',
    'Transaction Hour',
    'Account Age Days',
    'Address Mismatch',
    'Is Fraudulent',
    'Predicted Fraud',
    'Fraud Probability',
    'Risk Level'
]]

# Save to CSV
output.to_csv('fraud_predictions.csv', index=False)
print(f"Saved! Total rows: {len(output)}")
print(f"Actual fraud in sample: {output['Is Fraudulent'].sum()}")
print(f"Predicted fraud in sample: {output['Predicted Fraud'].sum()}")