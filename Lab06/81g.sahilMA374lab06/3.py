import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import to_datetime
import scipy.stats as stats
from scipy.special import ndtri


def qqplot(df,interval):
    companies = list(df.columns)
    for j in range(5):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        axes = [ax1,ax2,ax3,ax4]
        for i in range(4):
            df1 = pd.DataFrame()
            df1[companies[4*j+i]] = df.loc[:,companies[4*j+i]]
            df1= df1.sort_values(by=[companies[4*j+i]],ascending=True).reset_index(drop=True)
            df1['count'] = df1.index + 1
            df1['comparison_points'] = (df1[companies[4*j+i]] - df1[companies[4*j+i]].mean())/df1[companies[4*j+i]].std(ddof=0)
            df1['real_normal'] = ndtri(df1['count']/df1.shape[0])
            print(df1['comparison_points'])
            axes[i].scatter(df1['real_normal'],df1['comparison_points'])
            axes[i].plot([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3], color='red')
            axes[i].set_title("Quantile Quantile Plot for {x} using {t} data".format(x = companies[4*j+i],t = interval))
            axes[i].set_xlabel("Theoretical Quantiles")
            axes[i].set_ylabel("Actual Quantiles")

        plt.tight_layout()
        plt.show()

def box_plot(df, interval):
    stocks_name = list(df.columns)
    df.fillna(method='ffill',inplace=True)
    df.fillna(method='bfill',inplace=True)
    for j in range(5):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        axes = [ax1,ax2,ax3,ax4]  
        for i in range(4):
            x = list(df[stocks_name[4*j + i]])
            y = []
            for k in range(len(x)-1):
                y.append(np.log(1 + (x[k+1]-x[k])/x[k]))

            axes[i].boxplot(y)
            axes[i].set_title(f"Boxplot for {stocks_name[4*j + i]} on {interval} basis")
        plt.tight_layout()
        plt.show()

             

def execute(filename):
    df = pd.read_csv(filename)
    stocks_name = list(df.columns)
    #daily
    df1 = df.copy()
    #weekly
    df2 = df.copy()
    df2 = df2.groupby(pd.DatetimeIndex(df2['Date']).to_period('W')).nth(0)
    #monthly
    df3 = df.copy()
    df3 = df3.groupby(pd.DatetimeIndex(df3['Date']).to_period('M')).nth(0)
    
    df1.set_index('Date', inplace=True)
    df2.set_index('Date', inplace=True)
    df3.set_index('Date', inplace=True)

    df7 = df1.copy()

    # df1= df1.sort_values(by=[companies[4*j+i]],ascending=True).reset_index(drop=True)
    #         df1['count'] = df1.index + 1
    #         df1['comparison_points'] = (df1[companies[4*j+i]] - df1[companies[4*j+i]].mean())/df1[companies[4*j+i]].std(ddof=0)
    #         df1['real_normal'] = ndtri(df1['count']/df1.shape[0])
    #         axes[i].scatter(df1['real_normal'],df1['comparison_points'])
    #         axes[i].plot([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3], color='red')
    stocks = list(df7.columns)
    ss = stocks[5]
    val = list(df7.loc[:, ss])
    np.sort(val)
    mean = np.nanmean(val)
    vol = np.nanstd(val, ddof=0)
    tot = len(val)
    idx = [(i+1)/tot for i in range(len(val))]
    th = ndtri(idx)
    cal = (val - mean)/(vol)
    print(cal[1000:])
    plt.scatter(th,cal)
    plt.plot([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3], color='red')
    plt.tight_layout()
    plt.show()


    # i = 0
    # for stock in stocks_name:
    #     if(stock == 'Date'):
    #         continue
    #     plt.rcParams["figure.figsize"] = (20, 5)
    #     x = np.linspace(- 3,3, 100)

    #     x1 = np.log(df1[stock]/df1[stock].shift(1))
    #     M = np.nanmean(x1)
    #     sigma = np.nanstd(x1)
    #     xx1 = x1
    #     x1 = (x1 - M)/(sigma)

    #     x2 = np.log(df2[stock]/df2[stock].shift(1))
    #     M = np.nanmean(x2)
    #     sigma = np.nanstd(x2)
    #     xx2 = x2
    #     x2 = (x2 - M)/(sigma)

    #     x3 = np.log(df3[stock]/df3[stock].shift(1))
    #     M = np.nanmean(x3)
    #     sigma = np.nanstd(x3)
    #     xx3 = x3
    #     x3 = (x3 - M)/(sigma)
        
    #     plt.subplot(2,2,1)
    #     plt.hist(x1, 40, density=True, color='orange', label='Normalized Returns')
    #     plt.plot(x, stats.norm.pdf(x, 0, 1), color = 'green', label = 'density function, N(0, 1)')
    #     plt.xlabel('Returns')
    #     plt.ylabel('Frequency')
    #     plt.title(f'Histogram of normalzed log returns for {stock} on daily basis')
    #     plt.legend()

    #     plt.subplot(2,2,2)
    #     plt.hist(x2, 40, density=True, color='orange', label='Normalized Returns')
    #     plt.plot(x, stats.norm.pdf(x, 0, 1), color = 'green', label = 'density function, N(0, 1)')
    #     plt.xlabel('Returns')
    #     plt.ylabel('Frequency')
    #     plt.title(f'Histogram of normalzed log returns for {stock} on weekly basis')
    #     plt.legend()

    #     plt.subplot(2,2,3)
    #     plt.hist(x3, 40, density=True, color='orange', label='Normalized Returns')
    #     plt.plot(x, stats.norm.pdf(x, 0, 1), color = 'green', label = 'density function, N(0, 1)')
    #     plt.xlabel('Returns')
    #     plt.ylabel('Frequency')
    #     plt.title(f'Histogram of normalzed log returns for {stock} on monthly basis')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    #     i+=1

    # box_plot(df1,'daily')
    # box_plot(df2,'weekly')
    # box_plot(df3,'monthly')
    qqplot(df1,'daily')
    qqplot(df2,'weekly')
    qqplot(df3,'monthly')



execute('bsedata1.csv')
execute('nsedata1.csv')