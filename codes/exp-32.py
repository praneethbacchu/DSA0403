import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Sample housing dataset
df = pd.DataFrame({
    "house_size_sqft": [850, 1100, 1350, 1500, 1750, 2000, 2250, 2500, 2800, 3100, 1200, 1900],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 2, 3],
    "price": [180000, 215000, 255000, 290000, 330000, 375000, 410000, 460000, 515000, 570000, 235000, 355000]
})

# 2. Bivariate analysis
corr = df["house_size_sqft"].corr(df["price"])
print(f"Pearson Correlation (house_size_sqft vs price): {corr:.4f}")

# 3. Train linear regression model on selected feature
X = df[["house_size_sqft"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Intercept: ${model.intercept_:,.2f}")
print(f"Slope ($/sqft): ${model.coef_[0]:,.2f}")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:   {r2:.4f}")
