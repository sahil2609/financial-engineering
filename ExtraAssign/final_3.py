import math
import numpy as np
import pandas as pd
import yfinance as yf
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

def MH(stock_name):
    data = yf.download(stock_name, start="2016-01-01", end="2021-12-31")
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
    plt.title(f'{stock_name} Stock Price Prediction Using MCMC')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()

# Example usage
MH('AAPL')
