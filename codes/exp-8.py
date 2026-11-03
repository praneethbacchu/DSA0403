import numpy as np
import pandas as pd
from google.colab import files
import io

# Retrieve the previously uploaded file or ask for it again if missing
try:
    filename = list(uploaded.keys())[0]
    df_sales = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"Using previously uploaded file: {filename}")
except Exception:
    print("Please upload your e-commerce sales CSV file:")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df_sales = pd.read_csv(io.BytesIO(uploaded[filename]))

# Dynamic column detection with better prioritization
try:
    # Prioritize 'product' or 'item' specifically over generic 'name'
    prod_cols = [c for c in df_sales.columns if any(k in c.lower() for k in ['product', 'item'])]
    if not prod_cols:
        prod_cols = [c for c in df_sales.columns if 'name' in c.lower()]
    prod_col = prod_cols[0]
except IndexError:
    prod_col = df_sales.columns[0]

try:
    qty_col = [c for c in df_sales.columns if any(k in c.lower() for k in ['quantity', 'qty', 'sold', 'units'])][0]
except IndexError:
    qty_col = df_sales.select_dtypes(include=[np.number]).columns[0]

print(f"\nAnalyzing top products using Columns: Product='{prod_col}' and Quantity='{qty_col}'")

# Group by product and sum quantities to find top 5 best sellers
top_products = df_sales.groupby(prod_col)[qty_col].sum().reset_index()
top_products = top_products.sort_values(by=qty_col, ascending=False).head(5)

print("\n=== TOP 5 BEST SELLING PRODUCTS ===")
for idx, row in enumerate(top_products.itertuples(), 1):
    print(f"{idx}. {getattr(row, prod_col)}: {getattr(row, qty_col)} units sold")
