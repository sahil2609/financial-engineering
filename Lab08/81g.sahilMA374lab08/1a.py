import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tabulate import tabulate

def compute(filename, num_of_months):
    index = 60 - num_of_months
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'],format="%d-%m-%Y")
    df_monthly = df.groupby(pd.DatetimeIndex(df.Date).to_period('M')).nth(0)
    # print(df_monthly.head)
    df_reduced = df_monthly.iloc[index :]
    # df_reduced.reset_index(inplace = True, drop = True) 
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

    stocks_name = list(df_reduced.columns)
    headers = ["Stock Name", "Volatility"]
    rows = [] 
    for i in range(len(stocks_name)):
        row = [stocks_name[i], std[i]]
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


# def compute1(filename):
#     df = pd.read_csv(filename)
#     index = 60 -1
#     df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
#     df_monthly = df.groupby(pd.DatetimeIndex(df['Date']).to_period('M')).nth(0)
#     df_reduced = df_monthly.iloc[index:]
#     idx = list(df.index[df['Date'] >= df_reduced.iloc[0]['Date']])
#     df_reduced = df.iloc[idx[0]:]
#     df_reduced = df_reduced.set_index('Date')
#     stocks = list(df_reduced.columns)
#     std = []
#     for s in stocks:
#         a1 = np.log(df_reduced[s]/df_reduced[s].shift(1))
#         std.append(np.nanstd(a1)*np.sqrt(252))
#     headers = ["Stock Name", "Volatility"]
#     rows = [] 
#     for i in range(len(stocks)):
#         row = [stocks[i], std[i]]
#         rows.append(row)
#     print(tabulate(rows, headers=headers, tablefmt="grid"))   


print("Volatility for BSE Data\n")
compute('bsedata1.csv',1)

print("Volatility for NSE Data\n")
compute('nsedata1.csv',1)