import io
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Step 1: Request CSV Upload
print("Please upload the A/B test CSV file (or press Cancel to use simulated conversion rate data):")
uploaded = files.upload()

# Setup realistic fallback simulated dataset (Design A vs Design B)
np.random.seed(101)
simulated_data = {
    'Design': ['Design A'] * 100 + ['Design B'] * 100,
    'Conversion_Rate': list(np.random.beta(a=2, b=18, size=100)) + list(np.random.beta(a=3.5, b=16.5, size=100))
}
df = pd.DataFrame(simulated_data)

if uploaded:
    try:
        file_name = list(uploaded.keys())[0]
        temp_df = pd.read_csv(io.BytesIO(uploaded[file_name]))
        
        # Identify Group and Conversion Rate columns dynamically
        group_col = None
        conversion_col = None
        
        for col in temp_df.columns:
            if any(k in col.lower() for k in ['design', 'group', 'version', 'variant', 'ab_test']):
                group_col = col
            elif any(k in col.lower() for k in ['rate', 'conversion', 'converted', 'revenue']):
                conversion_col = col
                
        # Strict validation: Check if we have two distinct groups with clinical/test sounding names
        is_valid_ab = False
        if group_col and conversion_col:
            unique_vals = [str(x).lower() for x in temp_df[group_col].dropna().unique()]
            if any(any(k in val for k in ['design', 'group', 'test', 'control', 'variant', 'version', 'a', 'b']) for val in unique_vals) and len(unique_vals) >= 2:
                is_valid_ab = True
            
        if is_valid_ab:
            df = temp_df[[group_col, conversion_col]].rename(columns={group_col: 'Design', conversion_col: 'Conversion_Rate'}).dropna()
            print(f"\nSuccessfully loaded: '{file_name}'")
        else:
            print("\nUploaded file did not contain typical A/B variant groups. Falling back to simulated e-commerce data.")
    except Exception as e:
        print(f"\nCould not parse uploaded file. Error: {e}. Falling back to simulated data.")
else:
    print("\nUsing simulated e-commerce A/B test dataset.")

# Ensure conversion rates are numeric
df['Conversion_Rate'] = pd.to_numeric(df['Conversion_Rate'], errors='coerce')
df = df.dropna()

# Group separation
groups = df['Design'].unique()
if len(groups) >= 2:
    group_a_label = groups[0]
    group_b_label = groups[1]
    
    a_data = df[df['Design'] == group_a_label]['Conversion_Rate']
    b_data = df[df['Design'] == group_b_label]['Conversion_Rate']
    
    # Display Group Statistics
    print("\n=== Group Summary Statistics ===")
    print(f"{group_a_label}: N={len(a_data)}, Mean={a_data.mean():.4f}, StdDev={a_data.std():.4f}")
    print(f"{group_b_label}: N={len(b_data)}, Mean={b_data.mean():.4f}, StdDev={b_data.std():.4f}")
    
    # 1. Test for equality of variances (Levene's Test)
    levene_stat, levene_p = stats.levene(a_data, b_data)
    equal_var = True if levene_p > 0.05 else False
    
    # 2. Perform independent two-sample t-test
    t_stat, p_val = stats.ttest_ind(a_data, b_data, equal_var=equal_var)
    
    print("\n=== Independent Two-Sample T-Test Results ===")
    print(f"Levene's test for equal variances p-value: {levene_p:.4f} (Equal variances assumed: {equal_var})")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.6f}")
    
    # 3. Conclusion
    alpha = 0.05
    print("\n=== Statistical Conclusion ===")
    if p_val < alpha:
        print(f"Since the p-value ({p_val:.6f}) is LESS than our significance level alpha (0.05),")
        print(f"we REJECT the null hypothesis. There is a STATISTICALLY SIGNIFICANT difference")
        print(f"in the mean conversion rates between {group_a_label} and {group_b_label}.")
    else:
        print(f"Since the p-value ({p_val:.6f}) is GREATER than our significance level alpha (0.05),")
        print(f"we FAIL TO REJECT the null hypothesis. There is NO statistically significant difference")
        print(f"in the mean conversion rates between {group_a_label} and {group_b_label}.")
        
    # 4. Plot distributions
    plt.figure(figsize=(8, 5))
    sns.kdeplot(a_data, fill=True, label=f"{group_a_label}", color='skyblue', alpha=0.5)
    sns.kdeplot(b_data, fill=True, label=f"{group_b_label}", color='salmon', alpha=0.5)
    plt.title('Distribution of Conversion Rates (A/B Test)')
    plt.xlabel('Conversion Rate')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print(f"Error: Analysis requires exactly 2 distinct groups. Found groups: {groups}")
