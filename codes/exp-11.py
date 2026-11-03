import io
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files

# 1. Prompt user to upload CSV file
print("Please upload your monthly sales CSV file:")
uploaded = files.upload()

# 2. Read the uploaded CSV file into a Pandas DataFrame
for filename in uploaded.keys():
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded '{filename}'!")

# Normalize column names to lowercase to handle potential mismatch issues
df.columns = [col.lower().strip() for col in df.columns]

# Handle the structure of your uploaded dataset ('order_date' and 'sales' columns)
if 'order_date' in df.columns and 'sales' in df.columns:
    # Clean and convert the sales column to numeric (remove potential formatting commas and coerce errors)
    df['sales'] = df['sales'].astype(str).str.replace(',', '', regex=False)
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    
    # Convert order_date to datetime using format='mixed' to handle mixed date formats safely
    df['order_date'] = pd.to_datetime(df['order_date'], format='mixed')
    
    # Group by Month and sum sales (drop NaN values from failed numeric conversion)
    monthly_df = df.dropna(subset=['sales', 'order_date'])
    monthly_df = monthly_df.groupby(monthly_df['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
    monthly_df['order_date'] = monthly_df['order_date'].astype(str)
    
    # Sort by month to ensure plots display chronological data
    monthly_df = monthly_df.sort_values('order_date')
    
    months = monthly_df['order_date']
    sales = monthly_df['sales']
    
    # Set up the visualization figure
    plt.figure(figsize=(18, 5))
    
    # Question 1: Line Plot
    plt.subplot(1, 3, 1)
    plt.plot(months, sales, marker='o', color='b', linestyle='-')
    plt.title('Monthly Sales Trend (Line Plot)')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.grid(True)
    plt.xticks(rotation=90)
    
    # Question 2: Scatter Plot
    plt.subplot(1, 3, 2)
    plt.scatter(months, sales, color='r', marker='x', s=100)
    plt.title('Monthly Sales Observations (Scatter Plot)')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.grid(True)
    plt.xticks(rotation=90)
    
    # Question 3: Bar Plot
    plt.subplot(1, 3, 3)
    plt.bar(months, sales, color='g', alpha=0.7)
    plt.title('Monthly Sales Volume (Bar Plot)')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.grid(axis='y', linestyle='--')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.show()
else:
    print("\nCould not find compatible date and sales columns. Please make sure 'order_date' and 'sales' exist.")
