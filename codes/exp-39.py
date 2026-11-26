import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Transaction dataset
df = pd.DataFrame({
    "customer_id": range(1, 16),
    "total_spent": [35, 42, 28, 45, 38, 120, 135, 115, 140, 125, 310, 340, 295, 360, 320],
    "items_purchased": [2, 3, 2, 3, 2, 7, 8, 6, 8, 7, 16, 18, 15, 19, 17]
})

X_scaled = StandardScaler().fit_transform(df[["total_spent", "items_purchased"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("Cluster Summary:")
print(df.groupby("cluster")[["total_spent", "items_purchased"]].mean().round(1))

# 2. Scatter plot visualization
plt.figure(figsize=(7, 5))
for c_id, group in df.groupby("cluster"):
    plt.scatter(group["items_purchased"], group["total_spent"], label=f"Cluster {c_id}", s=70)

plt.title("Customer Segments: Items Purchased vs. Total Spent")
plt.xlabel("Items Purchased")
plt.ylabel("Total Spent ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
