import numpy as np
import pandas as pd
from scipy import stats

# 1. Load the Excel file using Row 4 (index 3) as the column headers
file_path = r'C:\Users\prane\OneDrive\Desktop\dataset 22.xlsx'
df = pd.read_excel(file_path, header=3)

# 2. Clean the data (drop any missing values in the 'Rating' column)
ratings = pd.to_numeric(df['Rating'], errors='coerce').dropna()

# 3. Calculate sample statistics
n = len(ratings)
mean_rating = ratings.mean()
std_err = stats.sem(ratings)

# 4. Calculate the 95% Confidence Interval
confidence_level = 0.95
degrees_freedom = n - 1
ci_lower, ci_upper = stats.t.interval(
    confidence_level, df=degrees_freedom, loc=mean_rating, scale=std_err
)

# 5. Display the results
print(f"Product Category: {df['Product_Category'].iloc[0]}")
print(f"Sample Size (n): {n}")
print(f"Sample Mean Rating: {mean_rating:.2f}")
print(f"Standard Error: {std_err:.4f}")
print(f"{int(confidence_level*100)}% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
