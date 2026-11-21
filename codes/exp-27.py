import io
import pandas as pd
import numpy as np
from google.colab import files
from sklearn.linear_model import LogisticRegression

# 1. Load the already uploaded 'EXP_27.xlsx' file or prompt for upload
try:
    df_raw = pd.read_excel('/content/EXP_27.xlsx', header=None)
    print("Successfully loaded 'EXP_27.xlsx'")
except Exception as e:
    print("Could not find 'EXP_27.xlsx' in workspace. Please upload the file:")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df_raw = pd.read_excel(io.BytesIO(uploaded[filename]), header=None)

# Find the correct header row containing 'Customer_ID' or the actual columns
header_idx = None
for idx, row in df_raw.iterrows():
    row_str = row.astype(str).str.lower().values
    if any('customer_id' in str(val) or 'usage_minutes' in str(val) for val in row_str):
        header_idx = idx
        break

if header_idx is None:
    header_idx = 0

# Reload dataframe with correct header row
df = pd.read_excel('/content/EXP_27.xlsx', skiprows=header_idx)

# Drop empty columns and fully empty rows
df = df.dropna(how='all')

# Drop ID columns
id_cols = [col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()]
if id_cols:
    df = df.drop(columns=id_cols)

print(f"\nSuccessfully cleaned dataset!")

# Identify target (churn) and features
target_col = df.columns[-1]
feature_cols = [col for col in df.columns if col != target_col]

print(f"Target column (to predict): '{target_col}'")
print(f"Features used for training: {list(feature_cols)}")

# Convert to numeric, dropping NaNs
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(df[target_col], errors='coerce')

valid_rows = X.notna().all(axis=1) & y.notna()
X = X[valid_rows]
y = y[valid_rows]

# 2. Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
print("\nLogistic Regression model trained successfully on the uploaded dataset!")

# 3. Predict with a Sample Input
sample_customer = {}
for col in X.columns:
    if 'minute' in col.lower() or 'usage' in col.lower():
        sample_customer[col] = 450.0
    elif 'contract' in col.lower() or 'duration' in col.lower():
        sample_customer[col] = 12.0
    elif 'charge' in col.lower() or 'bill' in col.lower():
        sample_customer[col] = 55.0
    else:
        sample_customer[col] = float(X[col].mean())  # default to feature mean

sample_df = pd.DataFrame([sample_customer])[X.columns]

# Make prediction
prediction = model.predict(sample_df)[0]
probability = model.predict_proba(sample_df)[0]

print("\n--- Sample Prediction Run ---")
print("Sample Customer Specifications:")
for k, v in sample_customer.items():
    print(f"  - {k}: {v}")
    
if prediction == 1:
     print(f"\nPrediction: Customer is predicted to CHURN (1) with probability {probability[1]:.2%}.")
else:
     print(f"\nPrediction: Customer is predicted to NOT CHURN (0) with probability {probability[0]:.2%}.")
