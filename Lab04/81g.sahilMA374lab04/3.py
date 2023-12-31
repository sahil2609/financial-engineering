import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import io

def wweights(M,C,mu):
    u = [1 for i in range(len(M))]
    tp1 = np.linalg.det([[1, u@np.linalg.inv(C)@np.transpose(M)],[mu, M@np.linalg.inv(C)@np.transpose(M)]])
    tp2 = np.linalg.det([[ u@np.linalg.inv(C)@np.transpose(u), 1],[ M@np.linalg.inv(C)@np.transpose(u), mu]])
    tp3 = np.linalg.det([[ u@np.linalg.inv(C)@np.transpose(u), u@np.linalg.inv(C)@np.transpose(M)],[ M@np.linalg.inv(C)@np.transpose(u), M@np.linalg.inv(C)@np.transpose(M)]])

    w = (tp1*(u@np.linalg.inv(C)) + tp2*(M@np.linalg.inv(C)))/(tp3)
    return w


print("part a\n")
df = pd.read_csv('data.csv')
df.set_index('Date', inplace=True)
df = df.pct_change()
M = np.mean(df, axis = 0) * 12
C = df.cov()

returns = np.linspace(-3, 5, num = 5000)
u = np.array([1 for i in range(len(M))])
risk = []

for mu in returns:
    w = wweights(M, C, mu)
    sigma = math.sqrt(w @ C @ np.transpose(w))
    risk.append(sigma)
  
weight_min_var = u@np.linalg.inv(C) / (u @ np.linalg.inv(C) @ np.transpose(u))
mu_min_var = weight_min_var@np.transpose(M)
risk_min_var = math.sqrt(weight_min_var @ C @ np.transpose(weight_min_var))
returns_plot1, risk_plot1, returns_plot2, risk_plot2 = [], [], [], []
for i in range(len(returns)):
    if returns[i] >= mu_min_var: 
      returns_plot1.append(returns[i])
      risk_plot1.append(risk[i])
    else:
      returns_plot2.append(returns[i])
      risk_plot2.append(risk[i])

mu_rf = 0.05

market_portfolio_weights = (M - mu_rf * u) @ np.linalg.inv(C) / ((M - mu_rf * u) @ np.linalg.inv(C) @ np.transpose(u) )
mu_market = market_portfolio_weights@np.transpose(M)
risk_market = math.sqrt(market_portfolio_weights @ C @ np.transpose(market_portfolio_weights))

plt.plot(risk_plot1, returns_plot1, color = 'Orange', label = 'Efficient frontier')
plt.plot(risk_plot2, returns_plot2, color = 'Green')
plt.xlabel("Risk (sigma)")
plt.ylabel("Returns") 
plt.title("Minimum Variance Curve & Efficient Frontier")
plt.plot(risk_market, mu_market, color = 'green', marker = 'o')
plt.annotate('Market Portfolio (' + str(round(risk_market, 2)) + ', ' + str(round(mu_market, 2)) + ')', xy=(risk_market, mu_market), xytext=(0.2, 0.6))
plt.plot(risk_min_var, mu_min_var, color = 'green', marker = 'o')
plt.annotate('Minimum Variance Portfolio (' + str(round(risk_min_var, 2)) + ', ' + str(round(mu_min_var, 2)) + ')', xy=(risk_min_var, mu_min_var), xytext=(risk_min_var, -0.6))
plt.legend()
plt.show()

print("\npart b\n")
print("Market Portfolio Weights = ", market_portfolio_weights)
print("Return = ", mu_market)
print("Risk = ", risk_market*100, " %")

print("\npart c\n")
returns_cml = []
risk_cml = np.linspace(0, 2, num = 5000)
for i in risk_cml:
    returns_cml.append(mu_rf+(mu_market - mu_rf)* i/ risk_market)


slope, intercept = (mu_market - mu_rf) / risk_market, mu_rf
print("\nEquation of CML is:")
print("y = {:.2f} x + {:.2f}\n".format(slope, intercept))

###################################################################

plt.plot(risk, returns, color = 'Blue', label = 'Minimum Variance Curve')
plt.plot(risk_cml, returns_cml, color = 'Green', label = 'CML')
plt.title("Capital Market Line with Markowitz Efficient Frontier")
plt.xlabel("Risk (sigma)")
plt.ylabel("Returns") 
plt.legend()
plt.show()
##########################################################
plt.plot(risk_cml, returns_cml)
plt.xlabel("Risk (sigma)")
plt.ylabel("Returns") 
plt.title("Capital Market Line")
plt.show()
###################################################################

print("\npart d\n")
stocks_data = ['SBIN.NS', 'ASIANPAINT.NS', 'BHARTIARTL.NS', 'CIPLA.NS', 'IOC.NS', 'JSWSTEEL.NS', 'MARUTI.NS', 'WIPRO.NS', 'AXISBANK.NS', 'ONGC.NS']
beta_k = np.linspace(-1, 1, 5000)
mu_k = mu_rf + (mu_market - mu_rf)*beta_k
plt.plot(beta_k, mu_k)

print("Equation of Security Market Line is:")
print("mu = {:.2f} beta + {:.2f}".format(mu_market - mu_rf, mu_rf))

plt.title('Security Market Line for all the 10 assets')
plt.xlabel("Beta")
plt.ylabel("Mean Return")
plt.show()