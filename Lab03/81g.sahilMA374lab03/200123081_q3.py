import math
import numpy as np
import time
from tabulate import tabulate
import matplotlib.pyplot as plt

S0 = 100
T = 1.0
r = 0.08
sigma = 0.20

dp = {}

def markokv_helper(M,dt,q,u,d,Sm,maxSp,n):
    if n == M:
        return maxSp - Sm
    if dp.get((Sm, maxSp))!= None:
        return dp[Sm, maxSp]
    up = markokv_helper(M,dt,q,u,d,Sm*u, max(Sm*u, maxSp),n+1)
    dn = markokv_helper(M,dt,q,u,d,Sm*d, max(maxSp,Sm*d),n+1)
    dp[Sm, maxSp] = math.exp(-r*dt)*(q*up + (1-q)*dn)
    return dp[Sm, maxSp]

def markov_algo(M):
    delt = (T*1.0)/(M*1.0)
    u = math.exp(sigma*(delt**0.5) + (r - 0.5*(sigma**2))*delt)
    d = math.exp(-sigma*(delt**0.5) + (r - 0.5*(sigma**2))*delt) 
    M = int(M)
    q = (math.exp(r*delt) - d)/(u - d)
    dp.clear()
    price = markokv_helper(M,delt,q,u,d,S0,S0,0)
    return price


M = [5,10,15,20,25,50]
head = ["M", "Option Value", "Time taken"]
rows = []
for i in M:
    s = round(time.time(), 10)
    value = markov_algo(i)
    e = round(time.time(), 10)
    rows.append([i, value, e-s])

print(tabulate(rows,headers=head,tablefmt="grid"))

prices = []
for m in range(5, 31, 1):
    price = markov_algo(m)
    prices.append(price)
plt.plot(range(5, 31, 1), prices)
plt.title('Initial option price vs M')
plt.ylabel('Initial option price')
plt.xlabel('M')
plt.show()