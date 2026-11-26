import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Sample patient clinical dataset
df = pd.DataFrame({
    "age": [34, 58, 45, 62, 29, 51, 40, 67, 38, 55, 47, 31, 60, 43, 53, 36],
    "gender": ["F", "M", "F", "M", "F", "M", "F", "M", "M", "F", "F", "M", "F", "M", "M", "F"],
    "blood_pressure": [118, 148, 125, 155, 115, 142, 122, 160, 128, 145, 130, 120, 150, 132, 140, 119],
    "cholesterol": [180, 240, 195, 255, 175, 230, 190, 265, 200, 235, 205, 185, 245, 210, 225, 178],
    "outcome": ["Good", "Bad", "Good", "Bad", "Good", "Bad", "Good", "Bad",
                "Good", "Bad", "Good", "Good", "Bad", "Good", "Bad", "Good"]
})

X = df.drop("outcome", axis=1)
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Preprocessing + KNN pipeline
prep = ColumnTransformer([
    ("cat", OneHotEncoder(drop="if_binary"), ["gender"]),
    ("num", StandardScaler(), ["age", "blood_pressure", "cholesterol"])
])

knn = Pipeline([
    ("prep", prep),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])
knn.fit(X_train, y_train)

# 3. Predictions & evaluation
y_pred = knn.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, pos_label='Good'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label='Good'):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, pos_label='Good'):.4f}\n")

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred
print("Test Set Predictions:")
print(results.to_string(index=False))
