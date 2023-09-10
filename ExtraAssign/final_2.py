import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Download the stock data for Apple from Yahoo Finance
stock = yf.download('GOOGL', start='2016-01-01', end='2022-04-30')

# Check for missing values
print(stock.isnull().sum())

# Check for infinity values
print(np.isinf(stock).sum())

# Replace infinity values with the mean value of the non-infinity values
stock.replace([np.inf, -np.inf], np.nan, inplace=True)
stock.fillna(stock.mean(), inplace=True)

# Fill any remaining missing values with the last observed value
stock.fillna(method='ffill', inplace=True)

# Convert data to float64 type
stock = stock.astype('float64')

# Split the data into training and testing sets
train = stock.iloc[:len(stock)-252, :]
test = stock.iloc[len(stock)-252:, :]

# Create the Exponential Smoothing model
model = ExponentialSmoothing(train['Adj Close'], trend='add', seasonal='add', seasonal_periods=252)

# Fit the model to the training data
model_fit = model.fit()

# Make predictions on the testing data
y_pred = model_fit.forecast(steps=len(test))

y_pred.fillna(y_pred.mean(), inplace=True)
y_pred.fillna(method='ffill', inplace=True)
y_pred = y_pred.astype('float64')

y_pred = list(y_pred)
test['Adj Close'].replace([np.inf, -np.inf], np.nan, inplace=True)
test['Adj Close'].fillna(test['Adj Close'].mean(), inplace=True)
test['Adj Close'].fillna(method='ffill', inplace=True)
test['Adj Close'] = test['Adj Close'].astype('float64')
y = list(test['Adj Close'])

rmse = mean_squared_error(y, y_pred, squared=False)
print('RMSE:', rmse)

# Plot the predicted vs actual plot

plt.figure(figsize=(12,6))
plt.plot(test.index, y, label='Actual')
plt.plot(test.index,y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Predicted vs Actual Stock Prices')
plt.legend()
plt.show()
