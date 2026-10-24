import numpy as np
import pandas as pd
from google.colab import files
import io

print("Please upload your grocery checkout CSV file:")
uploaded = files.upload()

try:
    # Read the uploaded file
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded {filename}!")
except Exception as e:
    print(f"\nCould not upload file ({e}). Using representative grocery dataset instead.\n")
    # Fallback dataset matching typical grocery checkout structures
    data = {
        'Item': ['Apples', 'Milk', 'Bread', 'Cereal', 'Chicken', 'Egg Carton', 'Orange Juice'],
        'Price': [2.99, 3.49, 1.99, 4.49, 9.99, 3.29, 4.19],
        'Quantity': [3, 2, 1, 2, 1, 1, 2],
        'Discount (%)': [10.0, 0.0, 5.0, 15.0, 0.0, 0.0, 10.0],  # percentage values
        'Tax (%)': [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]           # percentage values
    }
    df = pd.DataFrame(data)

print("\nDataset preview:")
print(df.head(10))

# Find column names dynamically based on keywords
price_col = [c for c in df.columns if 'price' in c.lower()][0]
qty_col = [c for c in df.columns if 'quant' in c.lower() or 'qty' in c.lower()][0]
discount_col = [c for c in df.columns if 'discount' in c.lower() or 'disc' in c.lower()][0]
tax_col = [c for c in df.columns if 'tax' in c.lower()][0]

# Convert values to numpy arrays for arithmetic operations
prices = df[price_col].values
quantities = df[qty_col].values
discounts = df[discount_col].values
taxes = df[tax_col].values

# Calculate subtotals (Price * Quantity)
subtotals = prices * quantities

# Calculate discount amounts (Subtotal * (Discount Rate / 100))
discount_amounts = subtotals * (discounts / 100.0)

# Subtract discount to get discounted subtotals
discounted_subtotals = subtotals - discount_amounts

# Calculate tax amounts on the discounted total (Discounted Subtotal * (Tax Rate / 100))
tax_amounts = discounted_subtotals * (taxes / 100.0)

# Total cost per item
item_totals = discounted_subtotals + tax_amounts

# Calculate aggregate totals
total_subtotal = np.sum(subtotals)
total_discounts = np.sum(discount_amounts)
total_taxes = np.sum(tax_amounts)
grand_total = np.sum(item_totals)

print("\n--- Cashier Checkout Receipt ---")
print(f"Subtotal before discounts: ${total_subtotal:,.2f}")
print(f"Discounts applied:        -${total_discounts:,.2f}")
print(f"Taxes applied:            +${total_taxes:,.2f}")
print(f"---------------------------------")
print(f"Total Cost:               ${grand_total:,.2f}")
