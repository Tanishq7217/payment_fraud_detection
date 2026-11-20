import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load data (same as model.py)
print("Loading data...")
df1 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data.csv")
df2 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data_2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Feature engineering - identical to model.py
df['Address Mismatch'] = (df['Billing Address'] != df['Shipping Address']).astype(int)

# Keep human-readable labels for the dashboard BEFORE encoding to codes
categorical = ['Payment Method', 'Product Category', 'Device Used']
labels = {col: df[col].copy() for col in categorical}
for col in categorical:
    df[col] = df[col].astype('category').cat.codes

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

# Recreate the exact train/test split used in model.py
# (same row count, test_size and random_state, no stratify -> identical indices)
_, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
df['Split'] = 'Train'
df.loc[test_idx, 'Split'] = 'Test'

# Only export rows the model has NEVER seen, so dashboard metrics are honest
test = df.loc[test_idx].copy()
print(f"Test rows (unseen by model): {len(test)}")

print("Loading model...")
model = joblib.load('fraud_model.pkl')

print("Predicting...")
test['Predicted Fraud'] = model.predict(test[features])
test['Fraud Probability'] = model.predict_proba(test[features])[:, 1]

test['Risk Level'] = pd.cut(
    test['Fraud Probability'],
    bins=[-0.01, 0.3, 0.7, 1.0],
    labels=['LOW', 'MEDIUM', 'HIGH']
).astype(str)
test['Risk Order'] = test['Risk Level'].map({'LOW': 1, 'MEDIUM': 2, 'HIGH': 3})

# Confusion-matrix outcome per row - makes Power BI visuals trivial
def outcome(row):
    if row['Is Fraudulent'] == 1 and row['Predicted Fraud'] == 1:
        return 'True Positive'
    if row['Is Fraudulent'] == 0 and row['Predicted Fraud'] == 1:
        return 'False Positive'
    if row['Is Fraudulent'] == 1 and row['Predicted Fraud'] == 0:
        return 'False Negative'
    return 'True Negative'

test['Outcome'] = test.apply(outcome, axis=1)
test['Actual Class'] = test['Is Fraudulent'].map({1: 'Fraud', 0: 'Legitimate'})
test['Predicted Class'] = test['Predicted Fraud'].map({1: 'Fraud', 0: 'Legitimate'})
test['Address Status'] = test['Address Mismatch'].map({1: 'Mismatch', 0: 'Match'})

# Readable category labels instead of 0/1/2/3 codes
for col in categorical:
    test[col] = labels[col].loc[test.index]

# Bands for segment analysis (Order columns = "Sort by column" in Power BI)
test['Amount Band'] = pd.cut(
    test['Transaction Amount'],
    bins=[-1, 100, 250, 500, 1000, float('inf')],
    labels=['0-100', '100-250', '250-500', '500-1000', '1000+']
).astype(str)
test['Amount Band Order'] = test['Amount Band'].map(
    {'0-100': 1, '100-250': 2, '250-500': 3, '500-1000': 4, '1000+': 5})

test['Account Age Band'] = pd.cut(
    test['Account Age Days'],
    bins=[-1, 7, 30, 90, 180, float('inf')],
    labels=['0-7 days', '8-30 days', '31-90 days', '91-180 days', '180+ days']
).astype(str)
test['Account Age Band Order'] = test['Account Age Band'].map(
    {'0-7 days': 1, '8-30 days': 2, '31-90 days': 3, '91-180 days': 4, '180+ days': 5})

output = test[[
    'Transaction Date',
    'Transaction Hour',
    'Transaction Amount',
    'Amount Band', 'Amount Band Order',
    'Payment Method',
    'Product Category',
    'Device Used',
    'Customer Age',
    'Account Age Days',
    'Account Age Band', 'Account Age Band Order',
    'Address Status',
    'Is Fraudulent', 'Actual Class',
    'Predicted Fraud', 'Predicted Class',
    'Fraud Probability',
    'Risk Level', 'Risk Order',
    'Outcome',
]]

output.to_csv('fraud_predictions.csv', index=False)

tp = (output['Outcome'] == 'True Positive').sum()
fp = (output['Outcome'] == 'False Positive').sum()
fn = (output['Outcome'] == 'False Negative').sum()
print(f"Saved! Total rows: {len(output)}")
print(f"Actual fraud: {output['Is Fraudulent'].sum()}  |  Predicted fraud: {output['Predicted Fraud'].sum()}")
print(f"Precision: {tp / (tp + fp):.2%}  |  Recall: {tp / (tp + fn):.2%}")
