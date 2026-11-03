import io
import pandas as pd
from google.colab import files

# Step 1: Upload the CSV file
print("Please upload your dataset CSV file:")
uploaded = files.upload()

# Step 2: Load and process the uploaded CSV
if uploaded:
    # Get the file name
    file_name = list(uploaded.keys())[0]
    
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(io.BytesIO(uploaded[file_name]))
    
    print(f"\nSuccessfully loaded '{file_name}' with shape: {df.shape}")
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Step 3: Identify the target column
    # Try to find 'likes' first; otherwise fallback to 'AvgTemperature' or the first numeric column
    target_col = None
    for col in df.columns:
        if col.lower() == 'likes':
            target_col = col
            break
            
    if not target_col:
        for col in df.columns:
            if col.lower() == 'avgtemperature':
                target_col = col
                break
                
    if not target_col:
        # Fallback to the first numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]

    if target_col:
        print(f"\nCalculating the frequency distribution for the '{target_col}' column...")
        
        # Calculate frequency distribution
        frequency_distribution = df[target_col].value_counts().sort_index()
        
        # Convert to DataFrame for a cleaner presentation
        freq_df = frequency_distribution.reset_index()
        freq_df.columns = [target_col, 'Frequency']
        
        print(f"\nFrequency Distribution of {target_col}:")
        print(freq_df.head(20).to_string(index=False)) # Displaying top 20 rows
        if len(freq_df) > 20:
            print(f"... and {len(freq_df) - 20} more rows.")
    else:
        print("\nError: Could not find any numeric columns in the uploaded CSV to calculate a distribution.")
else:
    print("No file was uploaded.")
