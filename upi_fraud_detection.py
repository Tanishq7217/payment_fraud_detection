import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load both files
df1 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data.csv")
df2 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data_2.csv")

# Combine them
df = pd.concat([df1, df2], ignore_index=True)

print("=== BASIC INFO ===")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

print("\n=== FRAUD ANALYSIS ===")

# Average transaction amount - fraud vs not fraud
print("\nAverage Transaction Amount:")
print(df.groupby('Is Fraudulent')['Transaction Amount'].mean())

# Fraud by payment method
print("\nFraud by Payment Method:")
print(df.groupby('Payment Method')['Is Fraudulent'].mean().sort_values(ascending=False))

# Fraud by device
print("\nFraud by Device Used:")
print(df.groupby('Device Used')['Is Fraudulent'].mean().sort_values(ascending=False))

# Fraud by hour
print("\nFraud by Transaction Hour (top 5):")
print(df.groupby('Transaction Hour')['Is Fraudulent'].mean().sort_values(ascending=False).head())

# Billing vs Shipping mismatch
df['Address Mismatch'] = (df['Billing Address'] != df['Shipping Address']).astype(int)
print("\nFraud rate with address mismatch:")
print(df.groupby('Address Mismatch')['Is Fraudulent'].mean())