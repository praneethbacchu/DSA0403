import io
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files

# 1. Prompt user to upload the temperature and rainfall CSV file
print("Please upload your monthly temperature and rainfall CSV file:")
uploaded_weather = files.upload()

# 2. Read the uploaded file
for filename in uploaded_weather.keys():
    df_weather = pd.read_csv(io.BytesIO(uploaded_weather[filename]))
    print(f"\nSuccessfully loaded '{filename}'!")

# Normalize column names to lowercase to make mapping easier
df_weather.columns = [col.lower().strip() for col in df_weather.columns]

# Check for necessary columns
# Map columns based on your CSV's headers: 'month', 'tem', 'rain'
month_col = next((col for col in df_weather.columns if 'month' in col), None)
temp_col = next((col for col in df_weather.columns if 'tem' in col), None)
rain_col = next((col for col in df_weather.columns if 'rain' in col), None)

if month_col and temp_col and rain_col:
    # Since the dataset spans multiple years, let's group by Month to get the average climate profile
    monthly_summary = df_weather.groupby(month_col).agg({temp_col: 'mean', rain_col: 'mean'}).reset_index()
    
    # Mapping month numbers to standard abbreviations for better labels
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    monthly_summary['month_name'] = monthly_summary[month_col].map(month_map)
    
    months = monthly_summary['month_name']
    temperature = monthly_summary[temp_col]
    rainfall = monthly_summary[rain_col]
    
    plt.figure(figsize=(14, 5))
    
    # Question 1: Line plot of the monthly temperature data
    plt.subplot(1, 2, 1)
    plt.plot(months, temperature, marker='o', color='crimson', linestyle='-', linewidth=2)
    plt.title('Average Monthly Temperature (Line Plot)')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    
    # Question 2: Scatter plot of the monthly rainfall data
    plt.subplot(1, 2, 2)
    plt.scatter(months, rainfall, color='teal', marker='D', s=100)
    plt.title('Average Monthly Rainfall (Scatter Plot)')
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
else:
    print(f"\nError: Could not identify columns. Found columns: {list(df_weather.columns)}")
