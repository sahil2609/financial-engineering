import yfinance as yf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Download data for the last 5 years
S = 'AAPL'
data = yf.download(S, period="5y")

# Split data into training and testing sets
train_data = data[:-252] # 4 years of data
test_data = data[-252:] # 1 year of data

# Define function to fit and forecast using ARIMA model
def arima_model(train_data, test_data, order):
    train_prices = train_data['Adj Close'].values
    test_prices = test_data['Adj Close'].values
    history = [x for x in train_prices]
    predictions = []
    for t in range(len(test_prices)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_prices[t]
        history.append(obs)
    mse = mean_squared_error(test_prices, predictions)
    return mse, predictions

# Train and test ARIMA model
order = (1, 1, 1) # (p, d, q)
mse, price_predictions = arima_model(train_data, test_data, order)
print("MSE:", mse)

# Plot actual prices and predicted prices
import matplotlib.pyplot as plt
plt.plot(test_data.index, test_data['Adj Close'], label="Actual Prices")
plt.plot(test_data.index, price_predictions, label="Predicted Prices")
plt.legend()
plt.show()
