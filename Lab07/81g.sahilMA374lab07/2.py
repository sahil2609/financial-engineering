import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits import mplot3d


def plot_3D(x_list, t_list, call_prices, z_label, plt_title):
    x, y, z = [], [], []

    for idx1 in range(len(t_list)):
        for idx2 in range(len(x_list)):
            x.append(x_list[idx2])
            y.append(t_list[idx1])
            z.append(call_prices[idx1][idx2])
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, cmap='Green')
    plt.title(plt_title)
    ax.set_xlabel("s") 
    ax.set_ylabel("t") 
    ax.set_zlabel(z_label)
    plt.show()

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


############ question 2 #############

T = 1
K = 1
r = 0.05
sigma = 0.6
t = [0,0.2,0.4,0.6,0.8,1]
s = np.linspace(0.1,3,10000)

### 2D plot ####
C = []
P =[]
for i in t:

    C1 = []
    P1 =[]
    for j in s:
        c,p = calc_price(j,T,K,r,sigma,i)
        C1.append(c)
        P1.append(p)
    C.append(C1)
    P.append(P1)
for i in range(len(t)):
    plt.plot(s,C[i], label=f"t = {t[i]}")
plt.xlabel("s")
plt.ylabel("C(t,s)")
plt.title(f'Call Price vs s')   
plt.legend()
plt.show()
for i in range(len(t)):
    plt.plot(s,P[i],label=f"t = {t[i]}")
plt.xlabel("s")
plt.ylabel("P(t,s)")
plt.title(f'Put Price vs s')
plt.legend()
plt.show()

### 3D plot ###
plot_3D(s,t,C,"C(t, s)", "Dependence of C(t, s) on t and s")
plot_3D(s,t,P,"P(t, s)", "Dependence of P(t, s) on t and s")