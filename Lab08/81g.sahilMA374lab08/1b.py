import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import norm


def compute(filename):
    A = np.linspace(0.5,1.5,11)
    for i in range(len(A)):
        A[i] = round(A[i] , 4)
    def N(x):
        return norm.cdf(x)


    def calc_price(s,T, K, r, sigma, t):
        if(t == T):
            return max(0,s - K), max(0, K-s)
        t = T - t
        d1 = (math.log(s/K) + (r + (sigma**2)/2)*t)/(sigma*math.sqrt(t))
        d2 = d1 - sigma*math.sqrt(t)
        C = N(d1)*s - N(d2)*K*math.exp(-r*t)
        P = -N(-d1)*s + N(-d2)*K*math.exp(-r*t)

        return C,P

    index = 60 - 6
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'],format="%d-%m-%Y")
    df_monthly = df.groupby(pd.DatetimeIndex(df.Date).to_period('M')).nth(0)
    df_reduced = df_monthly.iloc[index :]
    df_reduced.reset_index(inplace = True, drop = True) 
    idx_list = df.index[df['Date'] >= df_reduced.iloc[0]['Date']].tolist()
    df_reduced = df.iloc[idx_list[0] :]
    df_reduced = df_reduced.set_index('Date')
    std  = []
    stocks = list(df.columns)
    stocks.pop(0)
    for i in stocks:
        x1 = np.log(df_reduced[i]/df_reduced[i].shift(1))
        s = np.nanstd(x1)
        std.append(s*np.sqrt(252))

    ### K = S0 ###
    df = pd.read_csv(filename)
    df.set_index('Date')
    stocks = list(df.columns)
    r = 0.05
    headers = ["Strike Price", "Call Price", "Put Price"]
    for i in range(1,len(stocks)):
        print(f"Prices for {stocks[i]} with volatility for 1 month = {std[i-1]}")
        rows = []

        for j in A:
            S0 = df.iloc[-1][stocks[i]]
            K = j*S0
            T = 0.5
            t = 0
            sigma = std[i-1]
            c,p = calc_price(S0,T,K,r,sigma,t)
            rows.append([f'{j}*S0', c,p])
        print(tabulate(rows,headers=headers,tablefmt="grid"))




### BSE Data ###
print("Calculation for BSE data\n")
compute('bsedata1.csv')
print("\n\n")

### BSE Data ###
print("Calculation for NSE data")
compute('nsedata1.csv')
