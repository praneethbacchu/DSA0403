import io
import pandas as pd
import numpy as np
from google.colab import files
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Upload and Clean Dataset
print("Please upload your Excel sheet (.xlsx or .xls) containing customer shopping features:")
uploaded = files.upload()

if not uploaded:
    print("No file uploaded. Please rerun the cell and upload your dataset.")
else:
    filename = list(uploaded.keys())[0]
    df_raw = pd.read_excel(io.BytesIO(uploaded[filename]), header=None)

    # Dynamically locate the header row
    header_idx = None
    for idx, row in df_raw.iterrows():
        row_str = row.astype(str).str.lower().values
        if any('customer_id' in str(val) or 'age' in str(val) or 'spending' in str(val) or 'score' in str(val) for val in row_str):
            header_idx = idx
            break

    if header_idx is None:
        header_idx = 0

    # Reload dataframe with correct headers
    df = pd.read_excel(io.BytesIO(uploaded[filename]), skiprows=header_idx)

    # Filter out empty rows and non-feature descriptive ID columns
    df = df.dropna(how='all')
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)

    # Keep only numeric columns for K-Means clustering
    X = df.select_dtypes(include=[np.number])
    
    # Drop any rows that still contain NaNs
    X = X.dropna()
    
    print(f"\nSuccessfully cleaned dataset '{filename}'!")
    print(f"Features detected for clustering: {list(X.columns)}")

    # 2. Setup K-Means Parameters
    try:
        num_clusters = int(input("Enter the number of segments/clusters (e.g., 3, 4, 5): "))
    except ValueError:
        print("Invalid input. Defaulting to 3 clusters.")
        num_clusters = 3

    # Standardize features before applying K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means Clustering Model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    print(f"\nSuccessfully trained K-Means with {num_clusters} clusters!")

    # 3. Predict/Assign Cluster with a Sample Input
    sample_customer = {}
    print("\n--- Running Sample Customer Prediction ---")
    for col in X.columns:
        if 'age' in col.lower():
            sample_customer[col] = 35.0
        elif 'income' in col.lower() or 'salary' in col.lower():
            sample_customer[col] = 65000.0
        elif 'score' in col.lower() or 'spend' in col.lower():
            sample_customer[col] = 50.0
        else:
            sample_customer[col] = float(X[col].mean())  # default to feature mean
            
    sample_df = pd.DataFrame([sample_customer])[X.columns]
    
    # Scale the sample feature vector using the fitted scaler
    sample_scaled = scaler.transform(sample_df)
    
    # Predict cluster assignment
    predicted_cluster = kmeans.predict(sample_scaled)[0]
    
    print("Sample Customer Specifications:")
    for k, v in sample_customer.items():
        print(f"  - {k}: {v}")
        
    print(f"\nPrediction: This customer is assigned to **Cluster {predicted_cluster}**.")
