import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Sample store transaction data aggregated by customer
df = pd.DataFrame({
    "customer_id": ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10"],
    "avg_transaction_amount": [25.0, 30.0, 140.0, 155.0, 65.0, 70.0, 28.0, 148.0, 62.0, 32.0],
    "visit_frequency": [2, 3, 15, 16, 8, 9, 2, 14, 7, 3]
})

X_scaled = StandardScaler().fit_transform(df[["avg_transaction_amount", "visit_frequency"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["spending_segment"] = kmeans.fit_predict(X_scaled)

print("Customer Segments:")
print(df.sort_values("spending_segment").to_string(index=False))

print("\nSegment Averages:")
print(df.groupby("spending_segment")[["avg_transaction_amount", "visit_frequency"]].mean().round(2))

