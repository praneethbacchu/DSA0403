import io
import pandas as pd
from google.colab import files

# 1. Prompt user to upload the sales CSV file containing customer ages
print("Please upload your sales CSV file containing customer age data:")
uploaded_sales = files.upload()

# 2. Process and analyze the uploaded CSV
for filename in uploaded_sales.keys():
    df_sales = pd.read_csv(io.BytesIO(uploaded_sales[filename]))
    print(f"\nSuccessfully loaded '{filename}'!")
    
    # Normalize column names to lowercase and strip whitespace for flexible matching
    df_sales.columns = [col.lower().strip() for col in df_sales.columns]
    
    # Try to find a column representing customer age or fallback to numerical scores/hours if uploaded
    age_col = next((col for col in df_sales.columns if 'age' in col), None)
    
    # Fallback columns if target dataset isn't standard
    if not age_col:
        age_col = next((col for col in df_sales.columns if col in ['scores', 'hours']), None)
    
    if age_col:
        # Clean selected column to ensure it contains numeric integers
        df_sales[age_col] = pd.to_numeric(df_sales[age_col], errors='coerce')
        
        # Drop any missing or NaN records
        df_ages = df_sales[age_col].dropna().astype(int)
        
        # Compute the frequency distribution
        age_distribution = df_ages.value_counts().sort_index()
        
        # Display the results
        print(f"\nFrequency Distribution (Column: '{age_col}'):")
        print("-" * 50)
        print(f"{'Value':<10} | {'Frequency (Count)':<30}")
        print("-" * 50)
        for val, count in age_distribution.items():
            print(f"{val:<10} | {count:<30}")
            
        # Total summary stats
        print("-" * 50)
        print(f"Total valid data points: {len(df_ages)}")
    else:
        print(f"\nError: Could not find an appropriate numerical column. Available columns: {list(df_sales.columns)}")
