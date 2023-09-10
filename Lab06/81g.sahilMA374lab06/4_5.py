import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import to_datetime


def plot_paths(x1,x2,x3,stock,df1,df2,df3,xx1):
    y1 = df1['Date'].to_list()
    y2 = df2['Date'].to_list()
    y3 = df3['Date'].to_list()
    xx2 = df2[stock].to_list()
    xx3 = df3[stock].to_list()

    plt.rcParams['figure.figsize'] = (20,5)
    plt.subplot(1,3,1)
    plt.plot(y1, x1, color= 'orange', label ='Predicted Stock Prices')
    plt.plot(y1, xx1, color= 'green', label ='Acutal Stock Prices')
    plt.xticks(np.arange(0, len(y1), int(len(y1)/4)), df1['Date'][0:len(y1):int(len(y1)/4)])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.title(f'Stock price comaprison for {stock} on daily basis')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(y2, x2, color= 'orange', label ='Predicted Stock Prices')
    plt.plot( y2, xx2, color= 'green', label ='Acutal Stock Prices')
    plt.xticks(np.arange(0, len(y2), int(len(y2)/4)), df1['Date'][0:len(y2):int(len(y2)/4)])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.title(f'Stock price comaprison for {stock} on weekly basis')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(y3, x3, color= 'orange', label ='Predicted Stock Prices')
    plt.plot(y3, xx3, color= 'green', label ='Acutal Stock Prices')
    plt.xticks(np.arange(0, len(y3), int(len(y3)/4)), df1['Date'][0:len(y3):int(len(y3)/4)])
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.title(f'Stock price comaprison for {stock} on monthly basis')

    plt.legend()
    plt.show()

def execute(filename):
    df = pd.read_csv(filename)
    stocks_name = df.columns
    #daily
    df1 = df.copy()
    #weekly
    df2 = df.copy()
    df2 = df2.groupby(pd.DatetimeIndex(df2['Date']).to_period('W')).nth(0)
    #monthly
    df3 = df.copy()
    df3 = df3.groupby(pd.DatetimeIndex(df3['Date']).to_period('M')).nth(0)

    df1_t = df1[(pd.to_datetime(df1["Date"]) <= '2021-12-31')]
    df2_t = df2[(pd.to_datetime(df2["Date"]) <= '2021-12-31')]
    df3_t = df3[(pd.to_datetime(df3["Date"]) <= '2021-12-31')]
    df1_p = df1[(pd.to_datetime(df1["Date"]) > '2021-12-31')]
    df2_p = df2[(pd.to_datetime(df2["Date"]) > '2021-12-31')]
    df3_p = df3[(pd.to_datetime(df3["Date"]) > '2021-12-31')]

    for stock in stocks_name:
        if(stock  == 'Date'):
            continue

        
        xx1 = list(df1[stock])
        x1 = np.log(df1_t[stock]/df1_t[stock].shift(1))
        s1 = np.nanstd(x1)
        v1 = np.var(x1)
        v1*= (len(x1) *252)/(len(x1) -1)
        m1 = np.nanmean(x1)*252 + 0.5*v1
        x2 = np.log(df2_t[stock]/df2_t[stock].shift(1))
        m2 = np.nanmean(x2)
        s2 = np.nanstd(x2)
        x3 = np.log(df3_t[stock]/df3_t[stock].shift(1)) 
        m3 = np.nanmean(x3)
        s3 = np.nanstd(x3)
        x1 = list(df1_t[stock])
        x2 = list(df2_t[stock])
        x3 = list(df3_t[stock])
        dt = 1
        S0 = x1[len(x1) - 1]
        for i in range(len(df1_p[stock])):
            Z = np.random.normal(0,1)
            S = S0*np.exp((m1 - 0.5*s1*s1)*(1/252) + np.sqrt(v1)*Z*np.sqrt(1/252))
            x1.append(S)
            S0 = S

        S0 = x2[len(x2) - 1]
        for i in range(len(df2_p[stock])):
            Z = np.random.normal(0,1)
            S = S0*np.exp((m2 - 0.5*s2*s2)*dt + s2*Z)
            x2.append(S)
            S0 = S

        S0 = x3[len(x3) - 1]
        for i in range(len(df3_p[stock])):
            Z = np.random.normal(0,1)
            S = S0*np.exp((m3 - 0.5*s3*s3)*dt + s3*Z)
            x3.append(S)
            S0 = S

        plot_paths(x1,x2,x3,stock,df1, df2, df3,xx1)


execute('bsedata1.csv')
execute('nsedata1.csv')