import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib.ticker import FormatStrFormatter

def plot3d(arr, options,df):
    for i in range(len(arr)):
        call, put, strike, days= [],[],[],[]
        exp = "Expiry" + arr[i]
        date = "Date" + arr[i]
        df[date] = pd.to_datetime(df[date],format="%d-%m-%Y")
        df[exp] = pd.to_datetime(df[exp],format="%d-%m-%Y")
        df["diff_days"] = (df[exp] - df[date])/np.timedelta64(1, 'D')
        option = options[i]
        for idx in range(len(df[f'Expiry{arr[i]}'])):
            r = random.random()
            if r <= 0.15:
                call.append(df.iloc[idx][f'Close_CE{arr[i]}'])
                put.append(df.iloc[idx][f'Close_PE{arr[i]}'])
                strike.append(df.iloc[idx][f'Strike Price{arr[i]}'])
                days.append(df.iloc[idx]['diff_days'])
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax1.scatter(strike, days, call, marker='.')
        ax2.scatter(strike, days, put, marker='.')
        ax1.set_title(f'3D plot: Call Option - {option}')
        ax1.set_xlabel(f'Strike Price')
        ax1.set_ylabel(f'Maturity (in days)')
        ax1.set_zlabel('Call Price')

        ax2.set_title(f'3D plot: Put Option - {option}')
        ax2.set_xlabel(f'Strike Price')
        ax2.set_ylabel(f'Maturity (in days)')
        ax2.set_zlabel('Put Price')

        plt.tight_layout()
        plt.show()

def plot2d(arr, options,df):
    for i in range(len(arr)):
        call, put, strike, days= [],[],[],[]
        exp = "Expiry" + arr[i]
        date = "Date" + arr[i]
        df[date] = pd.to_datetime(df[date],format="%d-%m-%Y")
        df[exp] = pd.to_datetime(df[exp],format="%d-%m-%Y")
        df["diff_days"] = (df[exp] - df[date])/np.timedelta64(1, 'D')
        option = options[i]
        for idx in range(len(df[f'Expiry{arr[i]}'])):
            r = random.random()
            if r <= 0.05:
                call.append(df.iloc[idx][f'Close_CE{arr[i]}'])
                put.append(df.iloc[idx][f'Close_PE{arr[i]}'])
                strike.append(df.iloc[idx][f'Strike Price{arr[i]}'])
                days.append(df.iloc[idx]['diff_days'])
        fig,ax = plt.subplots(2,2)
        axes = []
        for x in range(2):
            for y in range(2):
                axes.append(ax[x,y])

        axes[0].scatter(strike, call, marker='.')
        axes[0].set_title(f'Call Price vs Strike Price for {option}')
        axes[0].set(xlabel='Strike Price', ylabel='Call Price')
        axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[1].scatter(strike, put, marker='.')
        axes[1].set_title(f'Put Price vs Strike Price for {option}')
        axes[1].set(xlabel='Strike Price', ylabel='Put Price')
        axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[2].scatter(days, call, marker='.')
        axes[2].set_title(f'Call Price vs Maturity for {option}')
        axes[2].set(xlabel='Maturity (in days)', ylabel='Call Price')
        axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[3].scatter(days, put, marker='.')
        axes[3].set_title(f'Put Price vs Maturity for {option}')
        axes[3].set(xlabel='Maturity (in days)', ylabel='Put Price')
        axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.tight_layout()
        plt.show()    

tp = []
tp.append("")
k = []
k.append("NIFTY")
companies = ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "WIPRO.NS"]
df = pd.read_csv('NIFTYoptiondata.csv')
plot3d(tp, k,df)
plot2d(tp, k,df)

dfs = pd.read_excel("stockoptiondata.xlsx", sheet_name=None)
for c in companies:
    df = dfs[c]
    k = []
    k.append(c)
    plot3d(tp, k,df)
    plot2d(tp, k,df)