import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import to_datetime

def plot_stock_prices(filemane):
    df = pd.read_csv(filemane)
    # df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    #daily
    df1 = df.copy()
    # df1['Date'] = pd.to_datetime(df1['Date'], format="%d-%m-%Y")
    stock_prices = list(df1.columns)
    #weekly
    df2 = df.copy()
    # df2['Date'] = pd.to_datetime(df2['Date'], format="%d-%m-%Y")
    df2 = df2.groupby(pd.DatetimeIndex(df2['Date']).to_period('W')).nth(0)
    #monthly
    df3 = df.copy()
    # df3['Date'] = pd.to_datetime(df3['Date'], format="%d-%m-%Y")
    df3 = df3.groupby(pd.DatetimeIndex(df3['Date']).to_period('M')).nth(0)
    
    
    for stock in stock_prices:
        plt.rcParams["figure.figsize"] = (20, 5)
        if(stock == 'Date'):
            continue


        x1 = df1['Date'].to_list()
        x2 = df2['Date'].to_list()
        x3 = df3['Date'].to_list()

        y1 = df1[stock].to_list()
        y3 = df3[stock].to_list()
        y2 = df2[stock].to_list()

        plt.subplot(1, 3, 1)
        plt.plot(x1, y1)
        plt.xticks(np.arange(0, len(x1), int(len(x1)/3)), df1['Date'][0:len(x1):int(len(x1)/3)])
        plt.title('Plot for Stock prices for {} on daily basis'.format(stock))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(1, 3, 2)
        plt.plot(x2, y2)
        plt.xticks(np.arange(0, len(x2), int(len(x2)/3)), df2['Date'][0:len(x2):int(len(x2)/3)])
        plt.title('Plot for Stock prices for {} on weekly basis'.format(stock))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        plt.plot(x3, y3)
        plt.xticks(np.arange(0, len(x3), int(len(x3)/3)), df3['Date'][0:len(x3):int(len(x3)/3)])
        plt.title('Plot for Stock prices for {} on monthly basis'.format(stock))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


plot_stock_prices('bsedata1.csv')

plot_stock_prices('nsedata1.csv')
