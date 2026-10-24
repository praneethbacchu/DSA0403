import numpy as np
import pandas as pd
from google.colab import files
import io

print("Please upload your quarterly sales CSV file:")
uploaded = files.upload()

try:
    # Read the uploaded file
    filename = list(uploaded.keys())[0]
    df_sales = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded {filename}!")
    
    # Attempt to locate numeric column containing sales
    numeric_cols = df_sales.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 1:
        # Extract first 4 elements representing Q1, Q2, Q3, and Q4
        sales_data = df_sales[numeric_cols[0]].values[:4]
    else:
        raise ValueError("No numeric column found representing sales figures.")
except Exception as e:
    print(f"\nCould not read the uploaded file ({e}). Using representative quarterly sales data instead.\n")
    # Fallback to representative quarterly sales data (Q1, Q2, Q3, Q4)
    sales_data = np.array([120000, 135000, 150000, 180000])

print("Quarterly Sales Array:", sales_data)

# 1. Calculate total sales for the year using np.sum
total_sales = np.sum(sales_data)

# 2. Calculate percentage increase from Q1 (index 0) to Q4 (index 3)
q1_sales = sales_data[0]
q4_sales = sales_data[3]
percentage_increase = ((q4_sales - q1_sales) / q1_sales) * 100

print("\n--- Sales Performance Summary ---")
print(f"Total Sales for the Year: ${total_sales:,.2f}")
print(f"Q1 Sales:                 ${q1_sales:,.2f}")
print(f"Q4 Sales:                 ${q4_sales:,.2f}")
print(f"Percentage Increase (Q1 to Q4): {percentage_increase:.2f}%")
