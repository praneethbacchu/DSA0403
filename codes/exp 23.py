import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 1. Load the Excel file and automatically find the true header row
file_path = r'C:\Users\prane\OneDrive\Desktop\dataset 23.xlsx'
raw_df = pd.read_excel(file_path, header=None)

# Scan the first 10 rows to find where the actual column headers sit (skipping title/description rows)
header_idx = 0
for idx in range(min(10, len(raw_df))):
    row_vals = raw_df.iloc[idx].dropna().astype(str).tolist()
    if len(row_vals) >= 2 and any(k in ' '.join(row_vals).lower() for k in ['group', 'patient', 'id', 'treatment', 'control', 'score', 'outcome']):
        header_idx = idx
        break

df = pd.read_excel(file_path, header=header_idx)
df.columns = df.columns.astype(str).str.strip()
print(f"Loaded from Row {header_idx + 1} | Columns found:", df.columns.tolist())

# 2. Automatically identify the Group column and the Numeric Measurement column
group_col = [c for c in df.columns if any(k in c.lower() for k in ['group', 'treatment', 'arm', 'type'])][0]

# Find the numeric column (excluding ID columns)
numeric_cols = [c for c in df.columns if c != group_col and 'id' not in c.lower()]
score_col = numeric_cols[0]

df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
df = df.dropna(subset=[group_col, score_col])

# Separate Control (Placebo) and Treatment (New Drug) groups
groups = df[group_col].unique()
control_label = next((g for g in groups if 'control' in str(g).lower() or 'placebo' in str(g).lower()), groups[0])
treatment_label = next((g for g in groups if g != control_label), groups[1])

control_data = df[df[group_col] == control_label][score_col]
treatment_data = df[df[group_col] == treatment_label][score_col]

# 3. Perform Independent Two-Sample t-test
t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
alpha = 0.05

print("\n--- Clinical Trial Hypothesis Test Results ---")
print(f"{control_label} (n={len(control_data)}): Mean = {control_data.mean():.2f}, SD = {control_data.std():.2f}")
print(f"{treatment_label} (n={len(treatment_data)}): Mean = {treatment_data.mean():.2f}, SD = {treatment_data.std():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.6f}")

if p_value < alpha:
    print("Conclusion: Statistically significant effect (Reject Null Hypothesis, p < 0.05).")
else:
    print("Conclusion: No statistically significant effect (Fail to Reject Null Hypothesis, p >= 0.05).")

# 4. Visualize the Data and p-value using Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))

box = ax.boxplot(
    [control_data, treatment_data],
    tick_labels=[str(control_label), str(treatment_label)],
    patch_artist=True,
    widths=0.4
)

# Color the boxes
for patch, color in zip(box['boxes'], ['#8ecae6', '#90be6d']):
    patch.set_facecolor(color)

# Overlay individual patient data points
np.random.seed(42)
for i, group_vals in enumerate([control_data, treatment_data], start=1):
    x_jitter = np.random.normal(i, 0.04, size=len(group_vals))
    ax.scatter(x_jitter, group_vals, color='black', alpha=0.5, s=22, zorder=3)

# Add significance bracket and p-value annotation above the boxes
y_max = max(control_data.max(), treatment_data.max())
y_min = min(control_data.min(), treatment_data.min())
y_range = y_max - y_min if y_max != y_min else 10
y_bracket = y_max + (y_range * 0.08)
h = y_range * 0.03

ax.plot([1, 1, 2, 2], [y_bracket, y_bracket + h, y_bracket + h, y_bracket], lw=1.5, color='black')
sig_text = f"p = {p_value:.4f} ({'Significant' if p_value < alpha else 'Not Significant'})"
ax.text(1.5, y_bracket + h + (y_range * 0.01), sig_text, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Labels and styling
ax.set_title(f'Clinical Trial Effectiveness: {control_label} vs. {treatment_label}', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel(score_col, fontsize=11)
ax.set_ylim(y_min - (y_range * 0.08), y_bracket + h + (y_range * 0.14))
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
