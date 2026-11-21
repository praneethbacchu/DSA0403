import io
import pandas as pd
from google.colab import files
from sklearn.neighbors import KNeighborsClassifier

# 1. Upload the Excel file
print("Please upload your Excel sheet (.xlsx or .xls) containing the patient dataset:")
uploaded = files.upload()

if not uploaded:
    print("No file uploaded. Please run the cell again and upload your Excel sheet.")
else:
    filename = list(uploaded.keys())[0]
    # Load full dataset first
    df_raw = pd.read_excel(io.BytesIO(uploaded[filename]), header=None)
    
    # Find the row containing 'Patient_ID' to set as header dynamically
    header_idx = 0
    for idx, row in df_raw.iterrows():
        if row.astype(str).str.contains('Patient_ID', case=False).any():
            header_idx = idx
            break
            
    # Reload dataframe with correct header row
    df = pd.read_excel(io.BytesIO(uploaded[filename]), skiprows=header_idx)
    
    # Clean columns: drop 'Patient_ID' or similar ID columns
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)
        
    # Drop any row containing all NaN values
    df = df.dropna(how='all')
    
    print(f"\nSuccessfully cleaned dataset '{filename}'!")
    print("Features & Labels detected:")
    
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    print(f"  Target column: '{target_col}'")
    print(f"  Feature columns: {list(feature_cols)}")
    
    # Cast features and target to correct numeric data types
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(df[target_col], errors='coerce')
    
    # Drop any remaining rows with missing values
    valid_indices = X.notna().all(axis=1) & y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # 2. Train the KNN Classifier
    # Prompt user for k
    try:
        k = int(input("\nEnter the number of neighbors (k) [e.g., 3 or 5]: "))
    except ValueError:
        print("Invalid input. Defaulting k to 3.")
        k = 3
        
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    print(f"KNN Classifier trained successfully with k={k}.")
    
    # 3. Input features for a new patient to predict
    print("\nPlease enter the values for the new patient's symptoms/features:")
    new_patient_features = []
    for feature in feature_cols:
        while True:
            try:
                val = float(input(f"  Value for '{feature}': "))
                new_patient_features.append(val)
                break
            except ValueError:
                print("  Invalid number. Please enter a valid numerical value.")
                
    # 4. Predict using the trained model
    prediction = knn.predict([new_patient_features])[0]
    probabilities = knn.predict_proba([new_patient_features])[0]
    
    print("\n--- Prediction Result ---")
    if prediction == 1:
        print(f"The patient is predicted to HAVE the medical condition (Class 1) with probability {probabilities[1]:.2f}.")
    else:
        print(f"The patient is predicted to NOT HAVE the medical condition (Class 0) with probability {probabilities[0]:.2f}.")
