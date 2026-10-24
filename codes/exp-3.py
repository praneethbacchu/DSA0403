import numpy as np
import pandas as pd
from google.colab import files
import io

print("Please upload your housing dataset CSV file:")
uploaded = files.upload()

try:
    # Read the uploaded file
    filename = list(uploaded.keys())[0]
    df_houses = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded {filename}!")
    
    # Attempt to locate columns dynamically
    bedroom_col = [c for c in df_houses.columns if 'bedroom' in c.lower() or 'beds' in c.lower()][0]
    sqft_col = [c for c in df_houses.columns if 'sqft' in c.lower() or 'square' in c.lower() or 'area' in c.lower()][0]
    price_col = [c for c in df_houses.columns if 'price' in c.lower() or 'cost' in c.lower() or 'sale' in c.lower()][0]
    
    selected_cols = [bedroom_col, sqft_col, price_col]
    house_data = df_houses[selected_cols].values
    print(f"Created house_data matrix with columns: {selected_cols}")
except Exception as e:
    print(f"\nCould not automatically parse the CSV ({e}). Using representative housing data instead.\n")
    # Fallback representative dataset
    # Columns: [Number of Bedrooms, Square Footage, Sale Price]
    house_data = np.array([
        [3, 1500, 250000],
        [5, 3000, 450000],
        [4, 2200, 320000],
        [6, 4200, 600000],
        [5, 3500, 520000],
        [2, 1100, 180000]
    ])

# 1. Create a boolean mask to filter houses with more than 4 bedrooms (index 0 of house_data)
filter_condition = house_data[:, 0] > 4

# 2. Apply mask to isolate those houses
houses_over_four_beds = house_data[filter_condition]

# 3. Extract the sale prices (index 2 of house_data)
sale_prices = houses_over_four_beds[:, 2]

# 4. Calculate the average sale price of these houses
if len(sale_prices) > 0:
    average_price = np.mean(sale_prices)
    print(f"\nNumber of houses with > 4 bedrooms: {len(sale_prices)}")
    print(f"Average sale price of houses with more than 4 bedrooms: ${average_price:,.2f}")
else:
    print("\nNo houses with more than 4 bedrooms were found in the dataset.")
