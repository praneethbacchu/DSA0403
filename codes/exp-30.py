import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Create sample used car dataset
data = {
    "mileage": [15000, 45000, 80000, 12000, 60000, 95000, 25000, 70000, 35000, 110000],
    "age": [2, 5, 8, 1, 6, 9, 3, 7, 4, 10],
    "brand": ["Toyota", "Honda", "Ford", "Toyota", "Hyundai", "Ford", "Honda", "Hyundai", "Toyota", "Ford"],
    "engine_type": ["Petrol", "Petrol", "Diesel", "Hybrid", "Petrol", "Diesel", "Hybrid", "Petrol", "Petrol", "Diesel"],
    "price": [24000, 17500, 11000, 27500, 14500, 9200, 22000, 13000, 19800, 7800]
}
df = pd.DataFrame(data)

X = df[["mileage", "age", "brand", "engine_type"]]
y = df["price"]

# 2. Build CART pipeline with preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["brand", "engine_type"]),
        ("num", "passthrough", ["mileage", "age"])
    ]
)

model = Pipeline([
    ("prep", preprocessor),
    ("cart", DecisionTreeRegressor(max_depth=4, random_state=42))
])
model.fit(X, y)

# 3. User input for a new car (using default values if run non-interactively)
try:
    mileage = float(input("Enter car mileage (e.g., 30000): ") or 30000)
    age = float(input("Enter car age in years (e.g., 4): ") or 4)
    brand = input("Enter brand (Toyota/Honda/Ford/Hyundai): ").strip() or "Toyota"
    engine_type = input("Enter engine type (Petrol/Diesel/Hybrid): ").strip() or "Petrol"
except EOFError:
    mileage, age, brand, engine_type = 30000.0, 4.0, "Toyota", "Petrol"

new_car = pd.DataFrame([{
    "mileage": mileage,
    "age": age,
    "brand": brand,
    "engine_type": engine_type
}])

predicted_price = model.predict(new_car)[0]
print(f"\nPredicted Car Price: ${predicted_price:,.2f}")

# 4. Display decision path for the new car
cart_step = model.named_steps["cart"]
feature_names = model.named_steps["prep"].get_feature_names_out()
x_transformed = model.named_steps["prep"].transform(new_car)
if hasattr(x_transformed, "toarray"):
    x_transformed = x_transformed.toarray()

node_indicator = cart_step.decision_path(x_transformed)
leaf_id = cart_step.apply(x_transformed)[0]
node_index = node_indicator.indices[
    node_indicator.indptr[0]:node_indicator.indptr[1]
]

print("\nDecision Path for New Car:")
for node_id in node_index:
    if leaf_id == node_id:
        print(f"  -> Leaf Node {node_id}: Predicted Value = ${cart_step.value[node_id][0][0]:,.2f}")
    else:
        feat = feature_names[cart_step.feature[node_id]]
        thresh = cart_step.threshold[node_id]
        val = x_transformed[0][cart_step.feature[node_id]]
        op = "<=" if val <= thresh else ">"
        print(f"  [Node {node_id}] {feat} ({val:.2f}) {op} {thresh:.2f}")
