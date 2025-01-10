import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
print("Dataset preview:")
print(df.head())

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

new_data = pd.DataFrame(
    [[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]],
    columns=X.columns 
)
new_data_scaled = scaler.transform(new_data)
predicted_price_usd = model.predict(new_data_scaled)[0]

usd_to_inr = 82 
predicted_price_inr = predicted_price_usd * usd_to_inr

print(f"\nPredicted House Price:")
print(f"USD: ${predicted_price_usd:.2f}")
print(f"INR: ₹{predicted_price_inr:.2f}")
