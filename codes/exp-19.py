import io
import numpy as np
import pandas as pd
import scipy.stats as stats
from google.colab import files

# Step 1: Request File Upload
print("Please upload the clinical trial CSV file (or press Cancel to use simulated trial data of 50 patients):")
uploaded = files.upload()

# Setup default simulated dataset of 50 patients (25 Drug, 25 Placebo)
np.random.seed(42)
simulated_data = {
    'Group': ['New Drug'] * 25 + ['Placebo'] * 25,
    'BP_Reduction': list(np.random.normal(loc=14.5, scale=5.2, size=25)) + list(np.random.normal(loc=3.2, scale=4.8, size=25))
}
df = pd.DataFrame(simulated_data)

if uploaded:
    try:
        file_name = list(uploaded.keys())[0]
        temp_df = pd.read_csv(io.BytesIO(uploaded[file_name]))
        
        # Identify Group column and Blood Pressure Reduction column dynamically
        group_col = None
        bp_col = None
        
        for col in temp_df.columns:
            if any(k in col.lower() for k in ['group', 'treatment', 'type']):
                group_col = col
            elif any(k in col.lower() for k in ['reduction', 'bp', 'blood_pressure', 'change', 'diff']):
                bp_col = col
        
        # Strict validation: Check if we actually found clinical-sounding treatment/group categories
        is_valid_clinical = False
        if group_col and bp_col:
            unique_vals = [str(x).lower() for x in temp_df[group_col].dropna().unique()]
            if any(any(k in val for k in ['drug', 'placebo', 'treatment', 'control']) for val in unique_vals):
                is_valid_clinical = True
                
        if is_valid_clinical:
            df = temp_df[[group_col, bp_col]].rename(columns={group_col: 'Group', bp_col: 'BP_Reduction'}).dropna()
            print(f"\nSuccessfully loaded: '{file_name}'")
        else:
            print("\nUploaded file did not contain clear drug/placebo treatment group data. Falling back to simulated clinical trial data.")
    except Exception as e:
        print(f"\nCould not parse uploaded file. Error: {e}. Falling back to simulated clinical trial data.")
else:
    print("\nUsing simulated clinical trial dataset (50 patients).")

# Ensure BP_Reduction is numeric
df['BP_Reduction'] = pd.to_numeric(df['BP_Reduction'], errors='coerce')
df = df.dropna()

# Step 2: Calculate Confidence Intervals
groups = df['Group'].unique()
print("\n=== 95% Confidence Intervals for Mean Blood Pressure Reduction ===\n")

for grp in groups:
    grp_data = df[df['Group'].astype(str).str.lower() == str(grp).lower()]['BP_Reduction']
    n = len(grp_data)
    
    if n < 2:
        print(f"Group '{grp}' does not have enough data points (N={n}).")
        continue
        
    mean = np.mean(grp_data)
    sem = stats.sem(grp_data)  # Standard error of the mean
    
    # Calculate 95% Confidence Interval using t-distribution
    ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
    
    print(f"Group: {grp} (N = {n})")
    print(f"  Mean Reduction: {mean:.2f} mmHg")
    print(f"  Standard Error: {sem:.2f}")
    print(f"  95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f}) mmHg\n")
