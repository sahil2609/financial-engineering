import numpy as np
import matplotlib.pylab as plt
import math


def min_var_portfolio(M,C):
    u = [1 for i in range(len(M))]
    min_weight_var = (u@np.linalg.inv(C))/(u@np.linalg.inv(C)@np.transpose(u))
    min_mu = M@np.transpose(min_weight_var)
    min_rate = np.sqrt(min_weight_var@C@np.transpose(min_weight_var))
    return min_rate, min_mu


def wweights(M,C,mu):
    u = [1 for i in range(len(M))]
    tp1 = np.linalg.det([[1, u@np.linalg.inv(C)@np.transpose(M)],[mu, M@np.linalg.inv(C)@np.transpose(M)]])
    tp2 = np.linalg.det([[ u@np.linalg.inv(C)@np.transpose(u), 1],[ M@np.linalg.inv(C)@np.transpose(u), mu]])
    tp3 = np.linalg.det([[ u@np.linalg.inv(C)@np.transpose(u), u@np.linalg.inv(C)@np.transpose(M)],[ M@np.linalg.inv(C)@np.transpose(u), M@np.linalg.inv(C)@np.transpose(M)]])

    w = (tp1*(u@np.linalg.inv(C)) + tp2*(M@np.linalg.inv(C)))/(tp3)
    return w



M = [0.1, 0.2, 0.15]
C = [[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]]

print("part a\n")
returns = np.linspace(0, 0.5, num = 10000)
risk = []

weights_10, return_10, risk_10 = [], [], []
weights_15, return_15 = [], []
weights_18, risk_18 = [], []

ct = 0
for mu in returns:
    w = wweights(M, C, mu)
    sigma = math.sqrt(w @ C @ np.transpose(w))
    risk.append(sigma)

    ct += 1
    if ct % 1000 == 0:
      weights_10.append(w)
      return_10.append(mu)
      risk_10.append(sigma*sigma)

    if abs(sigma - 0.15) < math.pow(10, -4.5):
      weights_15.append(w)
      return_15.append(mu)

risk_min_var, mu_min_var = min_var_portfolio(M, C)
returns_plot1, risk_plot1, returns_plot2, risk_plot2 = [], [], [], []

for i in range(len(returns)):
    if returns[i] >= mu_min_var: 
      returns_plot1.append(returns[i])
      risk_plot1.append(risk[i])
    else:
      returns_plot2.append(returns[i])
      risk_plot2.append(risk[i])

plt.plot(risk_plot1, returns_plot1, color = 'black', label = 'Efficient frontier')
plt.plot(risk_plot2, returns_plot2, color = 'brown')
plt.xlabel("Risk (sigma)")
plt.ylabel("Returns") 
plt.title("Minimum variance line along with Markowitz Efficient Frontier")
plt.plot(risk_min_var, mu_min_var, color = 'green', marker = 'o')
plt.annotate('Minimum Variance Portfolio (' + str(  round(risk_min_var, 2)) + ', ' + str(round(mu_min_var, 2)) + ')', xy=(risk_min_var, mu_min_var), xytext=(risk_min_var + 0.05, mu_min_var))
plt.legend()
plt.grid(True)
plt.show()
##############################################################################################################

print("\npart b\n")
print("Index\tweights\t\t\t\t\trisk\t\t\treturn\n")
for i in range(10):
    print("{}.\t{}\t{}\t{}".format(i + 1, weights_10[i], return_10[i], risk_10[i]))

##############################################################################################################
print("\npart c\n")
min_return, max_return = return_15[0], return_15[1]
min_return_weights, max_return_weights = weights_15[0], weights_15[1]

if min_return > max_return:
    min_return, max_return = max_return, min_return
    min_return_weights, max_return_weights = max_return_weights, min_return_weights

print("Minimum return = {}".format(min_return))
print("weights = {}".format(min_return_weights))
print("\nMaximum return = {}".format(max_return))
print("weights = {}".format(max_return_weights))

##############################################################################################################

print("\npart d\n")
given_return = 0.18
w = wweights(M, C, given_return)
minimum_risk = math.sqrt(w@C@np.transpose(w))
print("Minimum risk for 18% return = ", minimum_risk * 100, " %")
print("Weights = ", w)

##############################################################################################################
print("\npart e\n")
mu_rf = 0.1
u = np.array([1, 1, 1])

market_portfolio_weights = (M - mu_rf * u) @ np.linalg.inv(C) / ((M - mu_rf * u) @ np.linalg.inv(C) @ np.transpose(u) )
mu_market = market_portfolio_weights@np.transpose(M)
risk_market = math.sqrt(market_portfolio_weights @ C @ np.transpose(market_portfolio_weights))
print("Market Portfolio Weights = ", market_portfolio_weights)
print("Return = ", mu_market)
print("Risk = ", risk_market * 100 , " %")
returns_cml = []
risk_cml = np.linspace(0, 1, num = 10000)
for i in risk_cml:
    returns_cml.append(mu_rf + (mu_market - mu_rf) * i / risk_market)

slope, intercept = (mu_market - mu_rf) / risk_market, mu_rf
print("\nEquation of CML is:")
print("y = {:.2f} x + {:.2f}\n".format(slope, intercept))

plt.scatter(risk_market, mu_market, color = 'orange', linewidth = 3, label = 'Market portfolio')
plt.plot(risk, returns, color = 'brown', label = 'Minimum variance curve')
plt.plot(risk_cml, returns_cml, color = 'green', label = 'CML')
plt.xlabel("Risk (sigma)")
plt.ylabel("Returns") 
plt.title("Capital Market Line with Minimum variance curve")
plt.grid(True)
plt.legend()
plt.show()

##############################################################################################################
print("\npart f\n")
sigma = 0.1
mu_curr = (mu_market - mu_rf) * sigma / risk_market + mu_rf
weight_rf = (mu_curr - mu_market) / (mu_rf - mu_market)
weights_risk = (1 - weight_rf) * market_portfolio_weights

print("Risk =", sigma * 100, " %")
print("Risk-free weights =", weight_rf)
print("Risky Weights =", weights_risk)
print("Returns =", mu_curr)

sigma = 0.25
mu_curr = (mu_market - mu_rf) * sigma / risk_market + mu_rf
weight_rf = (mu_curr - mu_market) / (mu_rf - mu_market)
weights_risk = (1 - weight_rf) * market_portfolio_weights

print("\nRisk =", sigma * 100, " %")
print("Risk-free weights =", weight_rf)
print("Risky Weights =", weights_risk)
print("Returns =", mu_curr)