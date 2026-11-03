import numpy as np
import pandas as pd
from google.colab import files
import io

print("Please upload your real estate dataset (CSV format):")
uploaded = files.upload()

try:
    # Read the uploaded CSV file
    filename = list(uploaded.keys())[0]
    property_data = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded {filename}!")
except Exception as e:
    print(f"\nCould not load the file ({e}). Using representative property data instead.\n")
    # Fallback representative dataset
    data = {
        'property_id': ['P101', 'P102', 'P103', 'P104', 'P105', 'P106', 'P107'],
        'location': ['Downtown', 'Suburbs', 'Downtown', 'Suburbs', 'Uptown', 'Downtown', 'Uptown'],
        'bedrooms': [3, 5, 2, 4, 5, 3, 6],
        'area': [1500, 3200, 1100, 2400, 3500, 1600, 4200],
        'price': [250000, 450000, 180000, 320000, 520000, 260000, 600000]
    }
    property_data = pd.DataFrame(data)

print("\n--- Property Data Preview ---")
print(property_data.head())

# Dynamic Column Identification
try:
    loc_col = [c for c in property_data.columns if 'loc' in c.lower()][0]
    bed_col = [c for c in property_data.columns if any(k in c.lower() for k in ['bed', 'room'])][0]
    area_col = [c for c in property_data.columns if any(k in c.lower() for k in ['area', 'sq', 'feet', 'size'])][0]
    price_col = [c for c in property_data.columns if any(k in c.lower() for k in ['price', 'cost', 'list'])][0]
except IndexError as e:
    # Fallback to column positions if detection fails
    loc_col, bed_col, area_col, price_col = property_data.columns[1], property_data.columns[2], property_data.columns[3], property_data.columns[4]

# 1. The average listing price of properties in each location
avg_price_per_loc = property_data.groupby(loc_col)[price_col].mean()
print("\n=== 1. Average Listing Price by Location ===")
print(avg_price_per_loc.map('${:,.2f}'.format).to_string())

# 2. The number of properties with more than four bedrooms
more_than_four_beds = property_data[property_data[bed_col] > 4]
print(f"\n=== 2. Properties with More Than 4 Bedrooms ===")
print(f"Number of properties: {len(more_than_four_beds)}")

# 3. The property with the largest area
largest_property = property_data.loc[property_data[area_col].idxmax()]
print("\n=== 3. Property with the Largest Area ===")
print(largest_property.to_frame().T.to_string(index=False))
