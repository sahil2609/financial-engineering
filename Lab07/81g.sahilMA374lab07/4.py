import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits import mplot3d
from tabulate import tabulate

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

def calc_price2(s,T, K, r, sigma, t,i,j,k,l):
    if(i==0):
        s =k
    if(i==1):
        T =k
    if(i==2):
        K =k
    if(i==3):
        r =k
    if(i==4):
        sigma =k
    if(j==0):
        s =l
    if(j==1):
        T =l
    if(j==2):
        K =l
    if(j==3):
        r =l
    if(j==4):
        sigma =l
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
t = [0,0.2,0.4,0.6,0.8,1]
s22 = 2
s1 = np.linspace(0.0001, 2, 1000)
T1 = np.linspace(1, 5, 1000)
K1 = np.linspace(0.5,2,1000)
r1 = np.linspace(0.02, 1, 1000)
s = [0.4,0.8,1.2,1.6,2.0]
sigma1 = np.linspace(0.1, 1.1, 1000)
A = [s1,T1,K1,r1,sigma1]
tt = ["s","T","K","r","sigma"]
tp1 = np.zeros((5,5,1000))
tp2 = np.zeros((5,5,1000))

for k in range(5):
    for j in range(1000):
        c = np.zeros(5)
        p = np.zeros(5)
        c[1],p[1] = calc_price(s[k],T1[j],K,r,sigma,0.4)
        c[2],p[2] = calc_price(s[k],T,K1[j],r,sigma,0.4)
        c[3],p[3] = calc_price(s[k],T,K,r1[j],sigma,0.4)
        c[4],p[4] = calc_price(s[k],T,K,r,sigma1[j],0.4)
        for i in range(1,5):
            tp1[k][i][j] = c[i]
            tp2[k][i][j] = p[i]

for j in range(1,5):
    
    for i in range(5):
        plt.plot(A[j], tp1[i][j], label = f's = {s[i]}')
    plt.title(f"Plot for C(t,s) vs {tt[j]}")
    plt.xlabel(f'{tt[j]}')
    plt.ylabel('C(t,s)')
    plt.legend()
    plt.show()
    plt.close()

    for i in range(5):
        plt.plot(A[j], tp2[i][j], label = f's = {s[i]}')
    plt.title(f"Plot for P(t,s) vs {tt[j]}")
    plt.xlabel(f'{tt[j]}')
    plt.ylabel('P(t,s)')
    plt.legend()
    plt.show()
    plt.close()


for j in range(1,5):
    headers = [tt[j], "C(t,s)", "P(t,s)"]
    t1 = []
    t2 = []
    t3 = []
    data = []
    for i in range(1000):
        if(i%100 == 0):
            t4 = [A[j][i],tp1[2][j][i],tp2[2][j][i]]
            data.append(t4)
    
    print(tabulate(data,headers=headers,tablefmt="grid"))


### 3d graph ###

s1 = np.linspace(0.0001, 2, 100)
T1 = np.linspace(1, 5, 100)
K1 = np.linspace(0.5,2,100)
r1 = np.linspace(0.02, 1, 100)
sigma1 = np.linspace(0.1, 1.1, 100)
A = [s1,T1,K1,r1,sigma1]
tt = ["s","T","K","r","sigma"]
for i in range(5):
    for j in range(i+1,5):
        C1 = []
        P1 = []
        for k in A[i]:
            for l in A[j]:
                P =[]
                C=[]
                c,p = calc_price2(s22,T,K,r,sigma,0.4,i,j,k,l)
                C.append(c)
                P.append(p)
            C1.append(C)
            P1.append(P)
        s,t = np.meshgrid(A[i],A[j])
        C1 = np.array(C1)
        P1 = np.array(P1)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(s, t, C1, cmap="summer",linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel(tt[i])
        ax.set_ylabel(tt[j])
        ax.set_zlabel('C(t,s)')
        plt.title(f"C(t,s) vs {tt[i]} vs {tt[j]}")
        plt.show()
        plt.close()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(s, t, P1, cmap="summer",linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel(tt[i])
        ax.set_ylabel(tt[j])
        ax.set_zlabel('P(t,s)')
        plt.title(f"P(t,s) vs {tt[i]} vs {tt[j]}")
        plt.show()
        plt.close()
