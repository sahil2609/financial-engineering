import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import FormatStrFormatter


def compute(filename):

    def volatility_historical_price(index):
        index = 60 - index
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'],format="%d-%m-%Y")
        df_monthly = df.groupby(pd.DatetimeIndex(df.Date).to_period('M')).nth(0)
        df_reduced = df_monthly.iloc[index :]
        df_reduced.reset_index(inplace = True, drop = True) 
        idx_list = df.index[df['Date'] >= df_reduced.iloc[0]['Date']].tolist()
        df_reduced = df.iloc[idx_list[0] :]
        df_reduced = df_reduced.set_index('Date')
        std = []
        stocks = list(df.columns)
        stocks.pop(0)
        for i in stocks:
            x1 = np.log(df_reduced[i]/df_reduced[i].shift(1))
            s = np.nanstd(x1)
            std.append(s*np.sqrt(252))
        return std

    A = np.linspace(0.5,1.5,11)
    for i in range(len(A)):
        A[i] = round(A[i],4)
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

    ### K - S0 ###
    df = pd.read_csv(filename)
    df.set_index('Date')
    stocks = list(df.columns)
    r = 0.05
    rows1 = np.zeros((len(stocks),len(A),60))
    rows2 = np.zeros((len(stocks),len(A),60))
    time = []
    std_graph = []
    for k in range(1,61):
        time.append(k)
        std  = volatility_historical_price(k)
        std_graph.append(std)
        for i in range(1,len(stocks)):
            for j in range(len(A)):
                S0 = df.iloc[-1][stocks[i]]
                K = A[j]*S0
                T = 0.5
                t = 0
                sigma = std[i-1]
                c,p = calc_price(S0,T,K,r,sigma,t)
                rows1[i][j][k-1] = c
                rows2[i][j][k-1] = p

    fig,ax = plt.subplots(2,3)
    fig.suptitle("Hisorical Volatility vs Time Period")
    axes = []
    for x in range(2):
        for y in range(3):
            axes.append(ax[x,y])
    
    for i in range(6):
        tp =[]
        for j in range(60):
            tp.append(std_graph[j][i])
        axes[i].plot(time, tp)
        axes[i].set_title(f'{stocks[i+1]}')
        axes[i].set(xlabel='Length of time period (in months)', ylabel='Volatility')
        axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.show()
    
    for j in range(len(A)):
        fig,ax = plt.subplots(2,3)
        fig.suptitle(f"Call Option Price vs Time Period (A = {A[j]})")
        axes = []
        for x in range(2):
            for y in range(3):
                axes.append(ax[x,y])

        for i in range(6):
            axes[i].plot(time, rows1[i+1][j])
            axes[i].set_title(f'{stocks[i+1]}')
            axes[i].set(xlabel='Length of time period (in months)', ylabel='Call Option Price')
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.tight_layout()
        plt.show()

    for j in range(len(A)):
        fig,ax = plt.subplots(2,3)
        fig.suptitle(f"Put Option Price vs Time Period (A = {A[j]})")
        axes = []
        for x in range(2):
            for y in range(3):
                axes.append(ax[x,y])

        for i in range(6):
            axes[i].plot(time, rows2[i+1][j])
            axes[i].set_title(f'{stocks[i+1]}')
            axes[i].set(xlabel='Length of time period (in months)', ylabel='Put Option Price')
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.tight_layout()
        plt.show()


### BSE Data ###
compute('bsedata1.csv')

### BSE Data ###
compute('nsedata1.csv')
