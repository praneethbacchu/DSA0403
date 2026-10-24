import numpy as np
import pandas as pd
from google.colab import files
import io

print("Please upload your e-commerce orders CSV file:")
uploaded = files.upload()

try:
    # Read the uploaded file
    filename = list(uploaded.keys())[0]
    df_orders = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded {filename}!")
except Exception as e:
    print(f"\nCould not upload file ({e}). Using representative dataset instead.\n")
    # Fallback representative dataset matching the requested columns
    data_orders = {
        'Customer ID': ['C101', 'C102', 'C101', 'C103', 'C102', 'C101', 'C104'],
        'Order Date': ['2023-01-15', '2023-02-10', '2023-01-20', '2023-03-05', '2023-02-15', '2023-04-01', '2023-03-12'],
        'Product Name': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Mouse', 'Monitor', 'Keyboard'],
        'Order Quantity': [1, 2, 1, 1, 5, 2, 3]
    }
    df_orders = pd.DataFrame(data_orders)

print("\n--- Order Data Preview ---")
print(df_orders.head())

# Robust dynamic column name matching with fallbacks
try:
    cust_col = [c for c in df_orders.columns if any(k in c.lower() for k in ['customer', 'cust', 'rep', 'id'])][0]
except IndexError:
    cust_col = df_orders.columns[0]

try:
    date_col = [c for c in df_orders.columns if any(k in c.lower() for k in ['date', 'month', 'year', 'time'])][0]
except IndexError:
    date_col = df_orders.columns[1]

try:
    prod_col = [c for c in df_orders.columns if any(k in c.lower() for k in ['product', 'item', 'name'])][0]
except IndexError:
    prod_col = df_orders.columns[2]

try:
    qty_col = [c for c in df_orders.columns if any(k in c.lower() for k in ['quantity', 'qty', 'sold'])][0]
except IndexError:
    qty_col = df_orders.columns[3]

print(f"\nMatched Columns: Customer='{cust_col}', Date='{date_col}', Product='{prod_col}', Quantity='{qty_col}'")

# Ensure Date column is in datetime format
df_orders[date_col] = pd.to_datetime(df_orders[date_col], errors='coerce')

# 1. The total number of orders made by each customer type / rep
customer_orders = df_orders.groupby(cust_col).size()

# 2. The average order quantity for each product
product_avg_qty = df_orders.groupby(prod_col)[qty_col].mean()

# 3. The earliest and latest order dates
valid_dates = df_orders[date_col].dropna()
if not valid_dates.empty:
    earliest_date = valid_dates.min()
    latest_date = valid_dates.max()
    date_range_str = f"   - Earliest Order Date: {earliest_date.strftime('%Y-%m-%d')}\n   - Latest Order Date:   {latest_date.strftime('%Y-%m-%d')}"
else:
    date_range_str = "   - Date parsing unsuccessful (no valid dates found)."

print("\n=== ANALYSIS RESULTS ===")
print("\n1. Total Orders Made:")
print(customer_orders.to_string())

print("\n2. Average Order Quantity for Each Product:")
print(product_avg_qty.to_string())

print(f"\n3. Order Date Range:")
print(date_range_str)
