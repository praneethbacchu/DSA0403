import io
import numpy as np
import pandas as pd
import scipy.stats as stats
from google.colab import files

# Step 1: Upload the CSV file
print("Please upload 'rare_elements.csv' (or press Cancel to use standard simulation data of 100 elements):")
uploaded = files.upload()

# Setup default/fallback data: 100 random concentration measurements (mean=25.5 ppm, std=4.5)
np.random.seed(42)
fallback_data = np.random.normal(loc=25.5, scale=4.5, size=100)
df = pd.DataFrame({'Concentration': fallback_data})

if uploaded:
    try:
        file_name = list(uploaded.keys())[0]
        temp_df = pd.read_csv(io.BytesIO(uploaded[file_name]))
        
        # Dynamically identify the numeric column representing element concentration
        numeric_cols = temp_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Use first numeric column
            df = temp_df[[numeric_cols[0]]].rename(columns={numeric_cols[0]: 'Concentration'}).dropna()
            print(f"\nSuccessfully loaded: '{file_name}' with {len(df)} measurements.")
        else:
            print("\nNo numeric columns found in uploaded file. Using standard simulation data.")
    except Exception as e:
        print(f"\nCould not parse uploaded file. Error: {e}. Using standard simulation data.")
else:
    print("\nUsing standard simulation data of 100 rare element measurements.")

# Clean and collect concentrations as a NumPy array
concentrations = df['Concentration'].to_numpy()

# Step 2: Get user inputs
try:
    # Get sample size
    max_size = len(concentrations)
    sample_size_input = input(f"\nEnter sample size to draw (integer, 2 to {max_size}, default {min(30, max_size)}): ")
    n = int(sample_size_input) if sample_size_input.strip() else min(30, max_size)
    if n < 2 or n > max_size:
        raise ValueError("Out of bounds")
except ValueError:
    n = min(30, max_size)
    print(f"Invalid input or out of bounds. Defaulting to sample size: {n}")

try:
    # Get confidence level
    cl_input = input("Enter desired confidence level (e.g., 0.95 for 95%, default 0.95): ")
    confidence_level = float(cl_input) if cl_input.strip() else 0.95
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("Must be between 0 and 1")
except ValueError:
    confidence_level = 0.95
    print("Invalid input. Defaulting to confidence level: 0.95")

try:
    # Get desired level of precision (Margin of Error)
    precision_input = input("Enter desired precision level (maximum allowed margin of error, default 1.5): ")
    desired_precision = float(precision_input) if precision_input.strip() else 1.5
except ValueError:
    desired_precision = 1.5
    print("Invalid input. Defaulting to precision: 1.5")

# Step 3: Perform Point Estimation on a random sample of size n
sample = np.random.choice(concentrations, size=n, replace=False)
point_estimate = np.mean(sample)
sample_std = np.std(sample, ddof=1) # Sample standard deviation
sem = sample_std / np.sqrt(n)       # Standard error of the mean

# Step 4: Calculate Confidence Interval using t-distribution
margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=n-1) * sem
ci = (point_estimate - margin_of_error, point_estimate + margin_of_error)

# Step 5: Output results
print("\n" + "="*50)
print("=== RARE ELEMENT CONCENTRATION ANALYSIS RESULTS ===")
print("="*50)
print(f"Dataset Size: {len(concentrations)}")
print(f"Sample Size Chosen (n): {n}")
print(f"Point Estimate (Sample Mean): {point_estimate:.4f} ppm")
print(f"Sample Standard Deviation: {sample_std:.4f} ppm")
print(f"Standard Error (SEM): {sem:.4f} ppm")
print(f"{confidence_level*100}% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f}) ppm")
print(f"Actual Precision (Margin of Error): ±{margin_of_error:.4f} ppm")
print(f"Desired Precision: ±{desired_precision:.4f} ppm")

if margin_of_error <= desired_precision:
    print("\nSuccess: The actual margin of error meets your desired level of precision!")
else:
    suggested_n = int(np.ceil(((stats.t.ppf((1 + confidence_level) / 2, df=n-1) * sample_std) / desired_precision)**2))
    print(f"\nNotice: The actual margin of error (±{margin_of_error:.4f}) is wider than desired (±{desired_precision:.4f}).")
    print(f"To achieve the desired precision of ±{desired_precision:.4f} at {confidence_level*100}% confidence, ")
    print(f"you would need an estimated minimum sample size of n ≈ {suggested_n}.")
