import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import xgboost as xgb

print("Loading data...")
df1 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data.csv")
df2 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data_2.csv")
df = pd.concat([df1, df2], ignore_index=True)
print(f"Total rows: {len(df)}")

print("\nPreparing features...")

df['Address Mismatch'] = (df['Billing Address'] != df['Shipping Address']).astype(int)

#convert categorical columns to numbers

df['Payment Method'] = df['Payment Method'].astype('category').cat.codes
df['Product Category'] = df['Product Category'].astype('category').cat.codes
df['Device Used'] = df['Device Used'].astype('category').cat.codes

# Pick our features (inputs to the model)

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

X = df[features] #input
y = df['Is Fraudulent'] #output

print(f"Features selected: {features}")

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

print("\nTraining XGBoost model...")
print("(This may take 1-2 minutes, please wait...)")

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=19,  # handles imbalanced data (95% vs 5%)
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("Model trained!")

print("\nTesting model on unseen data...")
y_pred = model.predict(X_test)

print("\n=== MODEL RESULTS ===")
print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(f"Correctly identified NOT fraud: {cm[0][0]}")
print(f"Wrongly flagged as fraud:       {cm[0][1]}")
print(f"Missed fraud (dangerous!):      {cm[1][0]}")
print(f"Correctly caught fraud:         {cm[1][1]}")

import joblib

joblib.dump(model,'fraud_model.pkl')
print("\nModel saved as fraud_model.pkl!")