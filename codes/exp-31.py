import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Sample customer behavioral & demographic dataset
df = pd.DataFrame({
    "customer_id": range(101, 113),
    "age": [24, 27, 52, 48, 31, 22, 60, 55, 29, 35, 58, 25],
    "annual_spend": [420, 390, 2400, 2150, 950, 310, 2600, 2250, 1050, 1120, 2500, 450],
    "browsing_mins_per_wk": [180, 165, 45, 50, 120, 210, 40, 55, 130, 115, 48, 190],
    "purchase_frequency": [5, 4, 18, 16, 10, 3, 20, 17, 11, 12, 19, 6]
})

features = ["age", "annual_spend", "browsing_mins_per_wk", "purchase_frequency"]
X_scaled = StandardScaler().fit_transform(df[features])

# 2. Cluster customers using K-Means into 3 segments
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

# 3. Summarize segments for marketing
segment_summary = df.groupby("segment")[features].mean().round(1)
segment_summary["customer_count"] = df.groupby("segment")["customer_id"].count()

print("Customer Segment Profiles:")
print(segment_summary)
print("\nSample Segmented Customers:")
print(df[["customer_id", "annual_spend", "purchase_frequency", "segment"]].head(6).to_string(index=False))
