import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from google.colab import files

# Try to let the user upload, but we will fall back on the hospital's 18-adult dataset if column mismatch occurs.
print("Please upload the CSV file (or press Cancel to use standard sample data of 18 adults):")
uploaded = files.upload()

# Initialize default data representing a typical hospital sample of 18 adults
sample_data = {
    'Age': [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61],
    '%Fat': [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]
}
df = pd.DataFrame(sample_data)

if uploaded:
    try:
        file_name = list(uploaded.keys())[0]
        temp_df = pd.read_csv(io.BytesIO(uploaded[file_name]))
        
        # Identify columns dynamically
        age_col = None
        fat_col = None
        for col in temp_df.columns:
            if col.lower() == 'age':
                age_col = col
            elif 'fat' in col.lower() or '%' in col.lower():
                fat_col = col
        
        if age_col and fat_col:
            df = temp_df[[age_col, fat_col]].rename(columns={age_col: 'Age', fat_col: '%Fat'}).dropna()
            print(f"\nSuccessfully loaded your uploaded file: '{file_name}'")
        else:
            print("\nUploaded file didn't match. Using standard hospital sample data of 18 adults.")
    except Exception as e:
        print(f"\nCould not parse uploaded file. Using standard hospital sample data. Error: {e}")
else:
    print("\nUsing standard hospital sample data for 18 adults.")

# Ensure columns are numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['%Fat'] = pd.to_numeric(df['%Fat'], errors='coerce')
df = df.dropna()

# 1. Calculate Mean, Median, and Standard Deviation
stats_summary = pd.DataFrame({
    'Mean': df.mean(),
    'Median': df.median(),
    'Std Dev': df.std()
})
print("\n=== Descriptive Statistics (N = {}) ===".format(len(df)))
print(stats_summary.round(2))

# 2. Draw Boxplots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'], color='skyblue')
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['%Fat'], color='salmon')
plt.title('Boxplot of %Fat')
plt.tight_layout()
plt.show()

# 3. Draw Scatter Plot
plt.figure(figsize=(6, 5))
sns.scatterplot(x=df['Age'], y=df['%Fat'], color='purple', s=80)
sns.regplot(x=df['Age'], y=df['%Fat'], scatter=False, color='red', line_kws={'linestyle': '--'})
plt.title('Scatter Plot: Age vs %Fat')
plt.xlabel('Age')
plt.ylabel('%Fat')
plt.tight_layout()
plt.show()

# 4. Draw Q-Q Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
stats.probplot(df['Age'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Age')

plt.subplot(1, 2, 2)
stats.probplot(df['%Fat'], dist="norm", plot=plt)
plt.title('Q-Q Plot for %Fat')
plt.tight_layout()
plt.show()
