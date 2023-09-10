import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

# Define the stock symbol and time period
symbol = "AAPL"
start_date = "2016-01-01"
end_date = "2021-12-31"

# Get the stock data using yfinance
stock_data = yf.download(symbol, start=start_date, end=end_date)

# Get the closing prices
closing_prices = stock_data['Close'].values.reshape(-1, 1)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Create the training and testing data
train_data = scaled_data[:int(len(scaled_data) * 0.8), :]
test_data = scaled_data[int(len(scaled_data) * 0.8):, :]

# Define the number of time steps
time_steps = 60

# Create the training and testing data with time steps
train_X = []
train_y = []
for i in range(time_steps, len(train_data)):
    train_X.append(train_data[i-time_steps:i, 0])
    train_y.append(train_data[i, 0])
train_X, train_y = np.array(train_X), np.array(train_y)

test_X = []
test_y = []
for i in range(time_steps, len(test_data)):
    test_X.append(test_data[i-time_steps:i, 0])
    test_y.append(test_data[i, 0])
test_X, test_y = np.array(test_X), np.array(test_y)

# Reshape the data for LSTM input
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_X, train_y, epochs=30, batch_size=16)

# Make predictions on the testing data
predictions = model.predict(test_X)
predictions = scaler.inverse_transform(predictions)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - scaler.inverse_transform(test_y.reshape(-1, 1))) ** 2)))
print(f"RMSE: {rmse}")

# Plot the results

plt.figure(figsize=(16,8))
plt.plot(scaler.inverse_transform(test_y.reshape(-1, 1)), label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.legend()
plt.show()




