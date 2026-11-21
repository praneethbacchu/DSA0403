from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)
print("Decision Tree Classifier trained successfully on the Iris dataset.\n")

# 3. Allow user to input features of a new flower
print("Please enter the dimensions for the new Iris flower:")
new_flower_features = []
for feature in feature_names:
    while True:
        try:
            val = float(input(f"  Value for '{feature}': "))
            new_flower_features.append(val)
            break
        except ValueError:
            print("  Invalid number. Please enter a valid numerical value.")

# 4. Predict the species of the new flower
new_flower_features = np.array([new_flower_features])
prediction_id = clf.predict(new_flower_features)[0]
predicted_species = target_names[prediction_id]
probabilities = clf.predict_proba(new_flower_features)[0]

# 5. Output the result
print("\n--- Prediction Result ---")
print(f"The predicted species for the input dimensions is: **{predicted_species}**")
print("Class Probabilities:")
for name, prob in zip(target_names, probabilities):
    print(f"  - {name}: {prob * 100:.1f}%")
