import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load stock price data from Yahoo Finance
symbol = "GOOGL"
stock_data = yf.download(symbol, start="2016-05-01", end="2021-04-30")

# Calculate daily returns
stock_data["Returns"] = stock_data["Adj Close"].pct_change()

# Set frequency of date index
stock_data.index.freq = "D"

# Define train and test sets
train_set = stock_data.iloc[:len(stock_data)-252]
train_set.index.freq = "D"
test_set = stock_data.iloc[len(stock_data)-252:]
test_set.index.freq = "D"

# Define ARIMA model parameters
p = 3
d = 1
q = 2

# Train ARIMA model
model = ARIMA(train_set["Adj Close"], order=(p, d, q))
model_fit = model.fit()

# Predict on test set
y_pred_test = model_fit.forecast(steps=len(test_set))[0]

# Calculate mean squared error
mse = mean_squared_error(test_set["Adj Close"], y_pred_test)
print("Mean squared error:", mse)

# Predict on entire dataset
y_pred_all = model_fit.forecast(steps=len(stock_data))[0]

# Create date index for predicted prices
pred_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=len(y_pred_all), freq='D')

# Set frequency of date index for predicted prices
pred_dates.freq = "D"

# Plot actual vs predicted prices
plt.plot(stock_data.index, stock_data["Adj Close"], label="Actual prices")
plt.plot(pred_dates, y_pred_all, label="Predicted prices")
plt.axvline(x=train_set.index[-1], color="k", linestyle="--", label="Training set")
plt.axvline(x=test_set.index[0], color="k", linestyle="-.", label="Test set")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("ARIMA prediction for GOOGL stock prices")
plt.show()
