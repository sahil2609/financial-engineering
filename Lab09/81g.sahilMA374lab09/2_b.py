import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime
from scipy.misc import derivative
from scipy.stats import norm

def fx(C, x, K, t, T, r, sigma):
  if t == T:
    return max(0, x - K), max(0, K - x)

  d1 = ( math.log(x/K) + (r + 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )
  d2 = ( math.log(x/K) + (r - 0.5 * sigma * sigma) * (T - t) ) / ( sigma * math.sqrt(T - t) )

  call_price = x * norm.cdf(d1) - K * math.exp( -r * (T - t) ) * norm.cdf(d2)
  return C - call_price


def dfx(fx, var=0, point=[]):
  args = point[:]
  def wraps(x):
    args[var] = x
    return fx(*args)

  return derivative(wraps, point[var], dx = 1e-4)
  

def newton_method(epsilon, C, x, K, t, T, r, x0):
  itnum = 1
  root = 0

  while itnum <= 1000:
    denom = dfx(fx, 6, [C, x, K, t, T, r, x0])
    if denom == 0:
      return -1

    x1 = x0 - fx(C, x, K, t, T, r, x0)/denom
    if abs(x1 - x0) <= epsilon:
      root = x1
      break

    x0 = x1
    itnum += 1

  return root

r = 0.05
def fn(delay,s,k,p,sigma):
    if(delay == 0):
      return max(0, s-k)
    d1 = (np.log(s/k) + (r + (sigma**2)/2)*delay)/(sigma*np.sqrt(delay))
    d2 = d1 - sigma*(delay**0.5)
    num = s*norm.cdf(d1) - k*np.exp(-r*delay)*norm.cdf(d2) - p
    den = s*norm.pdf(d1)*(delay**0.5)
    return [num, den]


def newton_raphson(delay,s,k,p):
    sig = 0.3
    i = 0
    while(i<1000):
        x = fn(delay,s,k,p,sig)
        if(x[1] == 0):
            return -1
        tp = sig - (x[0]/x[1])
        if(abs(tp - sig) <= 1e-6):
            return tp
        sig = tp
        i+=1
    return sig


def plotd(arr, options,df,f=0):
    
    for i in range(len(arr)):
        option = options[i]
        stock_df = pd.read_csv('nsedata1.csv').loc[:,["Date",option]]
        df["Date"] = pd.to_datetime(df["Date"],format="%d-%m-%Y")
        stock_df["Date"] = pd.to_datetime(stock_df["Date"],format="%d-%m-%Y")
        df = pd.merge(df , stock_df, on="Date")
        call, put, strike, days, volatility= [],[],[],[],[]
        exp = "Expiry" + arr[i]
        date = "Date" + arr[i]
        df[date] = pd.to_datetime(df[date],format="%d-%m-%Y")
        df[exp] = pd.to_datetime(df[exp],format="%d-%m-%Y")
        df["diff_days"] = (df[exp] - df[date])/np.timedelta64(1, 'D')
        
        for idx in range(len(df[f'Expiry{arr[i]}'])):
            r = random.random()
            dee = 0.1
            if(f):
                dee = 0.01
            if r <= dee:
                if(int(df.iloc[idx][f'Close_CE{arr[i]}'])!=0 and df.iloc[idx]['diff_days']!=0):
                    sigma = newton_raphson(df.iloc[idx]['diff_days']/365,df.iloc[idx][f'Strike Price{arr[i]}'],df.iloc[idx][f'Strike Price{arr[i]}'],df.iloc[idx][f'Close_CE{arr[i]}'])
                    # sigma = newton_method(1e-6, df.iloc[idx][f'Close_CE{arr[i]}'], df.iloc[idx][f'{option}{arr[i]}'], df.iloc[idx][f'Strike Price{arr[i]}'], 0, df.iloc[idx]['diff_days']/365, 0.05, 0.6)
                    if(sigma != -1):
                        volatility.append(sigma)
                        call.append(df.iloc[idx][f'Close_CE{arr[i]}'])
                        put.append(df.iloc[idx][f'Close_PE{arr[i]}'])
                        strike.append(df.iloc[idx][f'Strike Price{arr[i]}'])
                        days.append(df.iloc[idx]['diff_days'])
                             
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(strike, days, volatility, marker='.')
        ax1.set_title(f'3D plot: Implied Volatility - {option}')
        ax1.set_xlabel(f'Strike Price') 
        ax1.set_ylabel(f'Maturity (in days)')
        ax1.set_zlabel('Implied Volatility')

        plt.tight_layout()
        plt.show()

        fig,(ax1,ax2) = plt.subplots(1,2)

        ax1.scatter(strike, volatility, marker='.')
        ax1.set_title(f'Implied Volatility vs Strike Price for {option}')
        ax1.set(xlabel='Strike Price', ylabel='Implied Volatility')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax2.scatter(days, volatility, marker='.')
        ax2.set_title(f'Implied Volatility vs Maturity for {option}')
        ax2.set(xlabel='Maturity (in days)', ylabel='Implied Volatility')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.tight_layout()
        plt.show()   

tp = []
tp.append("")
k = []
k.append("^NSEI")
df = pd.read_csv('NIFTYoptiondata.csv')
plotd(tp, k,df,1)
companies = ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "WIPRO.NS"]
dfs = pd.read_excel("stockoptiondata.xlsx", sheet_name=None)
for c in companies:
    df = dfs[c]
    k = []
    k.append(c)
    plotd( tp, k,df)
