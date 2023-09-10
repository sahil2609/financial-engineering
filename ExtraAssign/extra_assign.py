import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import matplotlib.dates as mdates
import pulp
import math
import random


# Define the stock symbol and time period
symbol = "AAPL"
ticker = symbol

# Get the stock data using yfinance
stock_data = yf.download(symbol, period='5y', interval='1d')
y_actual = stock_data[-252:]['Close'].values

def LSTM_model(stock_data):

    # Get the closing prices
    closing_prices = stock_data['Close'].values.reshape(-1, 1)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    # Create the training and testing data
    train_data = scaled_data[:int(len(scaled_data) * 0.8), :]
    test_data = scaled_data[(int(len(scaled_data)) - 252):, :]

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
    plt.title(f"{ticker} Stock Price Prediction using LSTM")
    plt.show()
    return rmse,predictions

def ExponentialSmoothing_model(stock):

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
    plt.title(f'{ticker} Stock Price Prediction Using Exponential Smoothing')
    plt.legend()
    plt.show()

    return rmse,y_pred
    
mu, sig, N = 1.1, 1, 100000
pts = []


def q(x):
    return (1 / (math.sqrt(2 * math.pi * sig ** 2))) * (math.e ** (-((x - mu) ** 2) / (2 * sig ** 2)))

def MCMC(n):
    r = np.zeros(1)
    p = q(r[0])
    pts = []

    for i in range(N):
        rn = r + np.random.uniform(-1, 1)
        pn = q(rn[0])
        if pn >= p:
            p = pn
            r = rn
        else:
            u = np.random.rand()
            if u < pn / p:
                p = pn
                r = rn
        pts.append(r)

    pts = random.sample(pts, len(pts))
    pts = np.array(pts)
    
    return pts

def MH(data):
    data = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
    data = data['Close']
    data = np.array(data)
    
    hist_data = data[-252:]
    data = data[:-252]
    
    stock_pred = []
    dt1 = data
    maturnity = 1
    volatility = 0.25
    risk_free = 0.1
    timestep = 1
    steps = 252
    delta_t = maturnity / steps
    i = 0
    stock_pred.append(dt1[-1])
    while timestep < steps:
        stock_price = stock_pred[-i]
        time_exp = maturnity - delta_t * timestep
        # Generate z_t using MCMC method
        pts = MCMC(N)
        stock_price = stock_price * math.exp(((risk_free - 0.5 * (
            math.pow(volatility, 2))) * delta_t + volatility * math.sqrt(delta_t) * pts[timestep + 5]))
        stock_pred.append(stock_price)
        i = i + 1
        timestep = timestep + 1
    rmse = math.sqrt(mean_squared_error(hist_data, stock_pred))
    print(f'RMSE MCMC: {rmse}')
    
    # plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(hist_data, label='Actual')
    plt.plot(stock_pred, label='Predicted')
    plt.legend()
    plt.title(f'{ticker} Stock Price Prediction Using MCMC')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()


    return rmse, stock_pred

def calculate_weights(rmse_1, rmse_2, rmse_3, y_pred_1, y_pred_2, y_pred_3, y_actual):
    preds = []
    mse = []   
    weight_lstm = 0.8
    weight_mcmc = 0.4
    weight_smooth = 0.4 
    # weights solver
    model = pulp.LpProblem('Optimal_weights', pulp.LpMinimize)
    # weights--->variables
    weight_lstm = pulp.LpVariable("weight_lstm", lowBound = 0, upBound=0.7)
    weight_mcmc = pulp.LpVariable("weight_mcmc", lowBound = 0, upBound=0.7)
    weight_smooth = pulp.LpVariable("weight_smooth", lowBound = 0, upBound=0.7)
    for i in range(len(y_actual)-62):
        preds.append(y_pred_2[i]*weight_lstm + y_pred_3[i+60]*weight_mcmc + y_pred_1[i+60]*weight_smooth)
        
    for i in range (len(y_actual)-62):
        mse.append(y_actual[i+60] - preds[i])
    # target function--->mean squared error

    mse = np.mean(mse)
    sum_w = weight_lstm + weight_mcmc + weight_smooth 

    model += mse
    model += sum_w <= 1.0
    model += sum_w >= 1.0

    pulp.LpSolverDefault.msg = 1

    # solve #
    model.solve()
    print('model solve')
    status = model.solve()
    print("Model status: ", pulp.LpStatus[status])
    print(model)

    weight_mcmc_f = weight_mcmc.varValue
    weight_smooth_f = weight_smooth.varValue
    weight_lstm_f = weight_lstm.varValue

    preds_final = []
    # Create final predictions from 3 methods
    for i in range(len(y_actual)-62):
        preds_final.append(y_pred_2[i]*weight_lstm_f + y_pred_3[i+60]*weight_mcmc_f + y_pred_1[i+60]*weight_smooth_f)
    preds_final = np.vstack(preds_final) 

    mse = []
    for i in range (len(y_actual)-62):
        mse.append(abs(y_actual[i+60] - preds_final[i]))
    mse = np.mean(mse)
    rmse = math.sqrt(mse)
    print(f'RMSE = {rmse}')
    print(f'LSTM weight: {weight_smooth_f}')
    print(f'MCMC weight: {weight_mcmc_f}')
    print(f'Exponetial Smoothing weight: {weight_lstm_f}')
    return rmse, weight_lstm_f, weight_smooth_f, weight_mcmc_f

stock_data1 = stock_data.copy()
stock_data2 = stock_data.copy()
stock_data3 = stock_data.copy()
print("Running Exponential Smoothing...")
rmse_ExponentialSmoothing, y_pred_1 = ExponentialSmoothing_model(stock_data1)
print(f"RMSE for Exponetial Smoothing: {rmse_ExponentialSmoothing}")
print("Running LSTM...")
rmse_LSTM, y_pred_2 = LSTM_model(stock_data2)
print(f"RMSE for LSTM: {rmse_LSTM}")
print("Running MCMC...")
rmse_MCMC, y_pred_3 = MH(stock_data3)
print(f"RMSE for MCMC: {rmse_MCMC}")

# Calculate the optimal weights and the new RMSE value
rmse, w1_exp_smoothing_value, w2_lstm_value, w1_mcmc = calculate_weights(rmse_ExponentialSmoothing, rmse_LSTM, rmse_MCMC, y_pred_1, y_pred_2, y_pred_3, y_actual)


# Calculate the weighted predicted values
y_pred_weighted = [(w1_exp_smoothing_value * y_pred_1[i+60]) + (w2_lstm_value * y_pred_2[i]) + (w1_mcmc * y_pred_3[i+60]) for i in range(len(y_actual)-60)]

# Plot the actual vs predicted values
plt.scatter(y_actual[60:], y_pred_weighted)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Weighted Predicted vs Actual values")
plt.show()


plt.figure(figsize=(12,6))
plt.plot(stock_data[(-252 + 60):].index, y_actual[60:], label='Actual')
plt.plot(stock_data[(-252 +60):].index,y_pred_weighted, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{ticker} Stock Price Prediction Using Exponential Smoothing')
plt.legend()
plt.show()