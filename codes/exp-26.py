import io
import pandas as pd
import numpy as np
from google.colab import files
from sklearn.linear_model import LinearRegression

# 1. Load the already uploaded 'EXP_26.xlsx' file
try:
    df_raw = pd.read_excel('/content/EXP_26.xlsx', header=None)
    print("Successfully loaded 'EXP_26.xlsx'")
except Exception as e:
    print("Could not find 'EXP_26.xlsx' in workspace. Please upload the file:")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df_raw = pd.read_excel(io.BytesIO(uploaded[filename]), header=None)

# Find the row containing 'House_ID' to set as header dynamically
header_idx = 0
for idx, row in df_raw.iterrows():
    if row.astype(str).str.contains('House_ID', case=False).any():
        header_idx = idx
        break

# Reload dataframe with correct header row
df = pd.read_excel('/content/EXP_26.xlsx', skiprows=header_idx)

# Remove columns containing 'ID' or non-feature/target descriptors
cols_to_drop = [col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower() and col != df.columns[-1]]
df = df.drop(columns=cols_to_drop)

# Drop fully empty rows
df = df.dropna(how='all')

# Last column is the target (Price_USD / Unnamed: 5)
target_col = df.columns[-1]
feature_cols = [col for col in df.columns if col != target_col]

# Cast everything to numeric
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(df[target_col], errors='coerce')

# Drop NaN values
valid_rows = X.notna().all(axis=1) & y.notna()
X = X[valid_rows]
y = y[valid_rows]

print(f"\nFeatures used for training: {list(X.columns)}")
print(f"Target column (to predict): '{target_col}'")

# 2. Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)
print("Linear Regression model trained successfully on the Excel dataset!")

# 3. Predict with a Sample Input
sample_house = {
    'Area_SqFt': 2000.0,
    'Bedrooms': 3.0,
    'Location_Score': 8.0,
    'House_Age_Yrs': 5.0
}

# Map our sample inputs to the exact order of features used during training
sample_input_df = pd.DataFrame([sample_house])[X.columns]
predicted_price = model.predict(sample_input_df)[0]

print("\n--- Sample Prediction Run ---")
print("Sample House Specifications:")
for k, v in sample_house.items():
    print(f"  - {k}: {v}")
print(f"Predicted Market Value: ${predicted_price:,.2f}")
