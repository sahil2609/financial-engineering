import yfinance as yf
import pandas as pd
import numpy as np

# Retrieving data
ticker = "GOOGL"
start_date = "2016-05-01"
end_date = "2021-04-30"
data = yf.download(ticker, start=start_date, end=end_date)
# Cleaning and preprocessing
data = data.dropna()
data = data["Adj Close"].to_frame()
data = data / data.max()

# Creating input features with rolling window approach
window_size = 20
for i in range(window_size):
    data[f"lag_{i+1}"] = data["Adj Close"].shift(i+1)
data = data.dropna()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Splitting data into training and testing sets
X = data.drop("Adj Close", axis=1)
y = data["Adj Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Random Forest Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting on test data and calculating mean squared error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
import matplotlib.pyplot as plt

# Plotting actual and predicted stock prices
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data.index, data["Adj Close"], label="Actual", color="blue")
ax.plot(y_test.index, y_pred, label="Predicted", color="orange")
ax.legend()
plt.show()
