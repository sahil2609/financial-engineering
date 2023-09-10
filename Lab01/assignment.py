import numpy as np
from math import exp
import matplotlib.pyplot as plt
from tabulate import tabulate

#data of the pricing
S0 = 100.0
K = 105.0
T = 5
r = 0.05
sigma = 0.30

#function to get the required plots
def binomial_pricing_model(M = [1, 5, 10, 20, 50, 100, 200, 400], option = "Call", show_plot = True, show_priceTable = False, show_timeTable = False):
	list_of_Prices = []
	for step in M:
		dt = T/step
		u = exp(sigma*(dt**0.5) + (r - 0.5*(sigma**2))*dt)
		d = exp(-sigma*(dt**0.5) + (r - 0.5*(sigma**2))*dt)
		dim = step + 1
		q = (exp(-r*dt) - d) / (u-d)

		Stock = np.zeros((dim,dim))
		Payoff = np.zeros((dim,dim))
		factor = np.zeros((dim,dim))
        #initialization
		Stock[0][0] = S0
		factor[0][0] = 1

		for j in range(1, dim):
			for i in range(j):
				Stock[i][j] = Stock[i][j-1]*u
				Stock[i+1][j] = Stock[i][j-1]*d
				factor[i][j] = factor[i][j-1]*u
				factor[i+1][j] = factor[i][j-1]*d
		Stock = np.round(Stock, decimals = 6)

		if option is "Call":
			for i in range(dim):
				Payoff[i][dim-1] = max(0, factor[i][dim-1]*S0 - K)
		else:
			for i in range(dim):
				Payoff[i][dim-1] = max(0, K - factor[i][dim-1]*S0)

		for j in range(dim-2, -1, -1):
			for i in range(j+1):
				Payoff[i][j] = exp(-r*dt)*(q*Payoff[i][j+1] + 
									(1-q)*Payoff[i+1][j+1])
		list_of_Prices.append(Payoff[0][0])

	if show_plot:
		X = M
		Y = list_of_Prices
		plt.xlabel("Number of Steps")
		if option is "Put":
			plt.ylabel("Price of Put Option")
		else :
			plt.ylabel("Price of Call Option")
		plt.title(option+" Option Pricing")
		plt.plot(X, Y)
		plt.show()

	if show_priceTable:
		print("\n\t"+option+" Price")
		table = []
		X = M
		Y = list_of_Prices
		for i in range(len(X)):
			table.append([X[i], Y[i]])
		print(tabulate(table, headers=["Step Size", "Option Price"]))
	elif show_timeTable:
		print("\n\t"+option+" Price")
		Tabulation = np.array([0, 0.50, 1, 1.50, 3, 4.5])
		X = [int(i/dt) for i in Tabulation]
		Y = [Payoff[i] for i in X]
		table = []
		for i in range(len(X)):
			table.append([X[i]/4, Y[i]])
		print(tabulate(table, headers=["Time Step", "Option Price"]))

#the default values of the show_plot = true, show_priceTable = false, show_timeTable = false
M = [1, 5, 10, 20, 50, 100, 200, 400]
binomial_pricing_model(M, "Call", show_plot=False, show_priceTable=True)
binomial_pricing_model(M, "Put", show_plot=False, show_priceTable=True)

M = range(5, 100, 5)
binomial_pricing_model(M, "Call")

M = range(1, 100, 1)
binomial_pricing_model(M, "Call")

M = range(5, 100, 5)
binomial_pricing_model(M, "Put")

M = range(1, 100, 1)
binomial_pricing_model(M, "Put")

M = [20]
binomial_pricing_model(M, "Call", show_plot=False, show_timeTable = True)
binomial_pricing_model(M, "Put", show_plot=False, show_timeTable = True)