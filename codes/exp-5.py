import numpy as np
import pandas as pd
from google.colab import files
import io

print("Please upload your car fuel efficiency CSV file:")
uploaded = files.upload()

try:
    # Read the uploaded file
    filename = list(uploaded.keys())[0]
    df_cars = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded {filename}!")
except Exception as e:
    print(f"\nCould not upload file ({e}). Using representative dataset instead.\n")
    data_cars = {
        'Car Model': ['Model A', 'Model B', 'Model C', 'Model D', 'Model E'],
        'Fuel Efficiency (mpg)': [24.5, 30.2, 28.0, 35.8, 42.1]
    }
    df_cars = pd.DataFrame(data_cars)

# Display the DataFrame
print(df_cars)

# Extract the fuel efficiency values as a NumPy array
# Match the fuel efficiency column dynamically
mpg_col = [c for c in df_cars.columns if 'efficiency' in c.lower() or 'mpg' in c.lower()][0]
fuel_efficiency = df_cars[mpg_col].values

# 1. Calculate the average fuel efficiency using np.mean
average_efficiency = np.mean(fuel_efficiency)

# 2. Determine percentage improvement between two models
# Comparing the first model (index 0) and the last model (index -1)
model_initial = fuel_efficiency[0]
model_new = fuel_efficiency[-1]
percentage_improvement = ((model_new - model_initial) / model_initial) * 100

print("\n--- Analysis Summary ---")
print(f"Average Fuel Efficiency: {average_efficiency:.2f} mpg")
print(f"Initial Model Efficiency: {model_initial:.2f} mpg")
print(f"New Model Efficiency: {model_new:.2f} mpg")
print(f"Percentage Improvement: {percentage_improvement:.2f}%")
