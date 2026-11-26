import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load a sample dataset (Breast Cancer classification dataset)
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("--- Dataset Loaded Successfully ---")
print(f"Available Features (first 5 shown): {list(data.feature_names[:5])} ... Total {len(data.feature_names)} features")
print(f"Target Variable Name: 'target' (classes: {list(data.target_names)})")

# Ask user for input
print("\n--- User Input Configuration ---")
feature_input = input("Enter feature names to use for evaluation separated by commas (or press Enter to use all features): ").strip()
target_input = input("Enter target variable name (or press Enter to use 'target'): ").strip()

# Determine features to use
if feature_input:
    selected_features = [f.strip() for f in feature_input.split(',') if f.strip() in df.columns]
    if not selected_features:
        print("No valid features found. Using all features instead.")
        selected_features = list(data.feature_names)
else:
    selected_features = list(data.feature_names)

# Determine target to use
selected_target = target_input if (target_input and target_input in df.columns) else 'target'

print(f"\nEvaluating model using features: {selected_features[:3]}... and target: '{selected_target}'")

# Prepare features (X) and target (y)
X = df[selected_features]
y = df[selected_target]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
# Use average='binary' if we have a binary classification task
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Display the metrics
print("\n=== Model Evaluation Metrics (Test Data) ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
