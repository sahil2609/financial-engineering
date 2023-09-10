import pandas as  pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tabulate import tabulate


def plot3d(x,y,z):
    x,y = np.meshgrid(x,y)
    z = np.array(z)
    fig,ax = plt.subplots(subplot_kw = {"projection":"3d"})
    surf = ax.plot_surface(x,y,z, cmap ="summer")
    fig.colorbar(surf)

df = pd.read_csv('bsedata1.csv')
index = 60 - 1
df_reduced = df[pd.to_datetime(df['Date'])  <= '2022-11-30']
# df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
# df_monthly = df.groupby(pd.DatetimeIndex(df['Date']).to_period('M')).nth(0)
# df_reduced = df_monthly.iloc[index:]
# idx = list(df.index[df['Date'] >= df_reduced.iloc[0]['Date']])
# df_reduced = df.iloc[idx[0]:]
df_reduced = df_reduced.set_index('Date')
print(df_reduced)
stocks = list(df_reduced.columns)
print(stocks)
for i in stocks:
    x = np.log(df_reduced[i]/df_reduced[i].shift(1))
    x = np.nanstd(x)
    print(x*np.sqrt(252))

