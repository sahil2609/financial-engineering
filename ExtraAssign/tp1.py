import yfinance as yf
import numpy as np
from scipy import stats as si

# Download data for the last 5 years
S = 'AAPL'
data = yf.download(S, period="5y")

# Define function to price Asian call option using binomial model
def asian_call_binomial(S, K, r, sigma, T, N):
    delta_t = T / N
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)
    prices = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            prices[j, i] = S * u**(i-j) * d**j
    payoffs = np.maximum(0, np.average(prices[1:], axis=0) - K)
    for i in range(N-1, -1, -1):
        payoffs = np.exp(-r * delta_t) * (p * payoffs[:-1] + (1-p) * payoffs[1:])
        if np.isnan(payoffs.mean()):
            payoffs.fill(0)
    price = payoffs[0]
    return price

# Split data into training and testing sets
train_data = data[:-252] # 4 years of data
test_data = data[-252:] # 1 year of data

# Calculate annualized return for training period
train_annual_return = (train_data['Adj Close'][-1] / train_data['Adj Close'][0]) ** (1/4) - 1

# Train model using binomial model
K = 150 # strike price
r = train_annual_return # annualized return
sigma = np.std(train_data['Adj Close'].pct_change()) * np.sqrt(252) # annualized volatility
T = 1 # time to maturity in years
N = 1000 # number of steps in binomial model
price_predictions = []
for i in range(len(test_data)):
    S = test_data['Adj Close'][i]
    price = asian_call_binomial(S, K, r, sigma, T, N)
    print(price)
    price_predictions.append(price)

# Calculate MSE between actual prices and predicted prices
actual_prices = test_data['Adj Close'].values
mse = np.mean((actual_prices - price_predictions)**2)
print("MSE:", mse)

# Plot actual prices and predicted prices
import matplotlib.pyplot as plt
plt.plot(test_data.index, actual_prices, label="Actual Prices")
plt.plot(test_data.index, price_predictions, label="Predicted Prices")
plt.legend()
plt.show()
