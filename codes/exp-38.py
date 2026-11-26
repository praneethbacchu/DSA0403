import numpy as np
import pandas as pd

# 1. Generate 365 daily temperature readings for 4 cities
rng = np.random.default_rng(42)
days = np.arange(365)

df = pd.DataFrame({
    "Chennai": 29 + 3 * np.sin(2 * np.pi * days / 365) + rng.normal(0, 1.2, 365),
    "Delhi": 25 + 11 * np.sin(2 * np.pi * (days - 80) / 365) + rng.normal(0, 2.5, 365),
    "Bengaluru": 23 + 3.5 * np.sin(2 * np.pi * (days - 50) / 365) + rng.normal(0, 1.5, 365),
    "Shimla": 14 + 7 * np.sin(2 * np.pi * (days - 90) / 365) + rng.normal(0, 2.0, 365),
})

# 2. Metrics per city
summary = pd.DataFrame({
    "mean_temp": df.mean(),
    "std_temp": df.std(),
    "temp_range": df.max() - df.min()
}).round(2)

highest_range_city = summary["temp_range"].idxmax()
most_consistent_city = summary["std_temp"].idxmin()

print("City Temperature Summary:")
print(summary)
print(f"\n1. Mean temperature per city:\n{summary['mean_temp'].to_string()}")
print(f"\n2. Standard deviation per city:\n{summary['std_temp'].to_string()}")
print(f"\n3. Highest temperature range: {highest_range_city} ({summary.loc[highest_range_city, 'temp_range']}°C)")
print(f"4. Most consistent temperature: {most_consistent_city} (Std Dev: {summary.loc[most_consistent_city, 'std_temp']}°C)")
