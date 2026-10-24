import numpy as np
import pandas as pd
from google.colab import files
import io

print("Please upload your local dataset file (CSV or XLSX format):")
uploaded = files.upload()

try:
    # Get the filename of the uploaded file
    filename = list(uploaded.keys())[0]
    
    # Dynamically read Excel or CSV based on file extension
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df_sales = pd.read_excel(io.BytesIO(uploaded[filename]))
    else:
        df_sales = pd.read_csv(io.BytesIO(uploaded[filename]))
        
    print(f"\nSuccessfully loaded {filename}!")

    # Dynamically select numeric columns and extract values
    numeric_cols = df_sales.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        sales_data = df_sales[numeric_cols].values
        print(f"Loaded sales matrix shape: {sales_data.shape}")
    else:
        raise ValueError("No numeric columns found in the uploaded file.")
except Exception as e:
    print(f"\nCould not read the uploaded file ({e}). Using representative 3x3 sales data instead.\n")
    # Fallback to representative 3x3 array
    sales_data = np.array([
        [10.50, 15.20, 20.00],
        [8.99, 12.49, 15.00],
        [30.00, 45.50, 50.00]
    ])

print("Sales Data Matrix:\n", sales_data)

# Calculate the overall average price of all products sold
average_price = np.mean(sales_data)

print(f"\nThe average price of all products sold is: ${average_price:.2f}")
