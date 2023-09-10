# Import libraries
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.pyplot as plt

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


T = 1
K = 1
r = 0.05
sigma = 0.6
t = np.linspace(0,1,100)
s = np.linspace(0.0001,2,100)
C1 = []
P1 = []
for i in t:
    for j in s:
        P =[]
        C=[]
        c,p = calc_price(j,T,K,r,sigma,i)
        C.append(c)
        P.append(p)
    C1.append(C)
    P1.append(P)

t,s = np.meshgrid(t,s)
C1 = np.array(C1)
P1 = np.array(P1)
fig,ax = plt.subplots(subplot_kw = {'projection' : '3d'})
ss = ax.plot_surface(t,s,C1,cmap='summer')
fig.colorbar(ss)
ax.set_xlabel('s')
ax.set_ylabel('t')
ax.set_zlabel('C(t,s)')
plt.title("C(t,s) vs s vs t")
plt.show()
plt.close()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(s, t, P1, cmap="summer",linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('s')
ax.set_ylabel('t')
ax.set_zlabel('P(t,s)')
plt.title("P(t,s) vs s vs t")
plt.show()
