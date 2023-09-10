import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import norm
from scipy.special import ndtri
from matplotlib.ticker import  FormatStrFormatter



x = np.arange(100)
y = np.linspace(1,50,100)
z = np.linspace(5,500,100)
ax = plt.axes(projection = '3d')
ax.scatter3D(x,y,z,cmap = 'summer')
ax.set_xlabel("sdfds")
ax.set_ylabel("sfsdfdssdds")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.show()
plt.close()
fig, ax1 = plt.subplots()
fig.suptitle("fdsfsd")
ax1.plot(x,y)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.show()
plt.close()