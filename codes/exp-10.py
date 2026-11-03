import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
import io

print("Please upload your monthly sales CSV file:")
uploaded = files.upload()

try:
    # Read the uploaded file
    filename = list(uploaded.keys())[0]
    df_sales = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded {filename}!")
except Exception as e:
    print(f"\nCould not upload file ({e}). Using representative monthly sales dataset instead.\n")
    # Fallback representative dataset
    data_sales = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Sales': [15000, 18000, 22000, 20000, 25000, 28000, 31000, 29000, 33000, 35000, 42000, 48000]
    }
    df_sales = pd.DataFrame(data_sales)

# Dynamic column mapping
try:
    month_col = [c for c in df_sales.columns if any(k in c.lower() for k in ['month', 'date', 'time'])][0]
except IndexError:
    month_col = df_sales.columns[0]

try:
    sales_col = [c for c in df_sales.columns if any(k in c.lower() for k in ['sales', 'revenue', 'amount', 'total'])][0]
except IndexError:
    sales_col = df_sales.select_dtypes(include=[np.number]).columns[0]

# Group by month to avoid duplicates and ensure a clean timeline if needed
monthly_data = df_sales.groupby(month_col)[sales_col].sum().reset_index()

# Set up matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Line Plot of Monthly Sales
axes[0].plot(monthly_data[month_col], monthly_data[sales_col], marker='o', color='b', linestyle='-', linewidth=2)
axes[0].set_title('Monthly Sales Performance (Line Plot)', fontsize=14)
axes[0].set_xlabel('Month', fontsize=12)
axes[0].set_ylabel('Sales ($)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].tick_params(axis='x', rotation=45)

# 2. Bar Plot of Monthly Sales
axes[1].bar(monthly_data[month_col], monthly_data[sales_col], color='skyblue', edgecolor='black')
axes[1].set_title('Monthly Sales Distribution (Bar Plot)', fontsize=14)
axes[1].set_xlabel('Month', fontsize=12)
axes[1].set_ylabel('Sales ($)', fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.6)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
