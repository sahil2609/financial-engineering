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
from tabulate import tabulate
import time
r = 0.05


def comp(s,K,tau,sig,p):
	d1 = (1/(sig*(tau*0.5)))*(math.log(s/K) + ((r+(sig*sig/2))*tau))
	d2 = d1 - (sig*math.sqrt(tau))
	num = (s*norm.cdf(d1)) - (K*math.exp(-r*tau)*norm.cdf(d2)) - p
	den = s*norm.pdf(d1)*(tau*0.5)
	return [num,den]

def newton_method(epsilon,p,s,K,tau):
    sig = 0.3
    for i in range(1000):
        x = comp(s,K,tau,sig,p)
        if(x[1]==0):
            return -1
        t = sig
        sig = sig - x[0]/x[1]
        #print(sig)
        if(abs(sig-t)<1e-6):
            return sig
    return sig

def cal_historical_volatility(df, date, expiry,company):
    idx = []
    delay = (expiry -date).days
    for i in range(df.shape[0]):
        if((date - df['Date'][i]).days >=0 and (date - df['Date'][i]).days <= delay):
            idx.append(i)
    df1 = df.iloc[idx, :]
    df1 = df1[company]
    df1.fillna(method = "ffill", inplace = True)
    if(df1.shape[0] <= 1):
        return -1
    A = []
    tp = np.array(df1)
    for i in range(len(tp) -1 ):
        A.append(np.log(tp[i+1]/tp[i]))
    return(np.nanstd(A)*math.sqrt(252))


def compare_volatility(arr, options,df, f= 0):


    for i in range(len(arr)):
        t1 = time.time()
        option = options[i]
        stock_df = pd.read_csv('nsedata1.csv').loc[:,["Date",option]]
        df["Date"] = pd.to_datetime(df["Date"],format="%d-%m-%Y")
        stock_df["Date"] = pd.to_datetime(stock_df["Date"],format="%d-%m-%Y")
        df = pd.merge(df , stock_df, on="Date")
        table = []
        lim = 1
        call, put, strike, days, implied_volatility, historical_volatility= [],[],[],[],[],[]
        exp = "Expiry" + arr[i]
        date = "Date" + arr[i]
        df[date] = pd.to_datetime(df[date],format="%d-%m-%Y")
        df[exp] = pd.to_datetime(df[exp],format="%d-%m-%Y")
        df["diff_days"] = (df[exp] - df[date])/np.timedelta64(1, 'D')
        ct =0 
        ct2 = 0
        ct1 = 0
        for idx in range(len(df[f'Expiry{arr[i]}'])):
            if(df["diff_days"][idx] > 1):
                ct2+=1
            r = random.random()
            dee = 0.1
            if(f):
                dee = 0.05
            if r <= dee:
                if(int(df.iloc[idx][f'Close_CE{arr[i]}'])!=0 and df.iloc[idx]['diff_days']!=0):
                    sigma = newton_method(1e-6, df.iloc[idx][f'Close_CE{arr[i]}'], df.iloc[idx][f'{option}{arr[i]}'], df.iloc[idx][f'Strike Price{arr[i]}'], df.iloc[idx]['diff_days']/365)
                    if(sigma == -1):
                        continue
                    r = random.random()
                    if(r <= 0.):
                        his_vol = cal_historical_volatility(stock_df,df['Date'][idx],df['Expiry'][idx],option)
                        if(his_vol == -1 or sigma == -1):
                            continue
                        
                        historical_volatility.append(his_vol)
                        implied_volatility.append(sigma)
                        call.append(df.iloc[idx][f'Close_CE{arr[i]}'])
                        put.append(df.iloc[idx][f'Close_PE{arr[i]}'])
                        strike.append(df.iloc[idx][f'Strike Price{arr[i]}'])
                        days.append(df.iloc[idx]['diff_days'])

                        if lim <= 20:
                            table.append([lim, df.iloc[idx][f'Close_CE{arr[i]}'],df.iloc[idx][f'{option}{arr[i]}'] , df.iloc[idx]['diff_days'], historical_volatility[-1], implied_volatility[-1]])
                            lim += 1
                        ct += 1
                ct1 +=1
        t2 = time.time()
        print(t2 -t1)
        print(ct2)
        print(ct)
        print(ct1)           
        print('\t\t\tFor {}\t\t\t'.format(option))
        print(tabulate(table, headers=['SI No.', 'Call Price', 'Stock Price (S0)', 'Maturity (in days)', 'Historical Volatility', 'Implied Volatility']))           
                    
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(strike, days, historical_volatility, marker='.',label= "historical volatility")
        ax1.scatter(strike, days, implied_volatility, marker='.',label= "implied volatility")
        ax1.set_title(f'3D plot: Comparison between historical and implied volatility- {option}')
        ax1.set_xlabel(f'Strike Price')
        ax1.set_ylabel(f'Maturity (in days)')
        ax1.set_zlabel('Volatility')
        ax1.set_zlim(-0.03, 1.2)
        plt.legend(loc = 'upper left')

        plt.tight_layout()
        plt.show()

        fig,(ax1,ax2) = plt.subplots(1,2)

        ax1.scatter(strike, implied_volatility, marker='.', label = 'implied volatility')
        ax1.scatter(strike, historical_volatility, marker='.', label = 'historical volatility')
        ax1.set_title(f'Volatility vs Strike Price for {option}')
        ax1.set(xlabel='Strike Price', ylabel='Implied Volatility')
        ax2.legend()
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax2.scatter(days, implied_volatility, marker='.', label = 'implied volatility')
        ax2.scatter(days, historical_volatility, marker='.', label = 'historical volatility')
        ax2.set_title(f'Volatility vs Maturity for {option}')
        ax2.set(xlabel='Maturity (in days)', ylabel='Implied Volatility')
        ax2.legend()
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.tight_layout()
        plt.show()  
        
        plt.scatter(historical_volatility, implied_volatility, marker = '.')
        plt.xlabel('Historical Volatility')
        plt.ylabel('Implied Volatility')
        plt.title(f'Historical vs Implied volatility for {option}')
        plt.ylim(-0.03, 1.2)
        plt.axis("square")
        plt.tight_layout()
        plt.show()

tp = []
tp.append("")
k = []
k.append("^NSEI")
df = pd.read_csv('NIFTYoptiondata.csv')
compare_volatility  ( tp, k,df,1)
companies = ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "WIPRO.NS"]
dfs = pd.read_excel("stockoptiondata.xlsx", sheet_name=None)
for c in companies:
    df = dfs[c]
    k = []
    k.append(c)
    compare_volatility(tp, k,df)
