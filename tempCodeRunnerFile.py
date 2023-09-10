import numpy as np
import math
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits import mplot3d
import pandas as pd


def comp_min(n,M,S0,u,d):
    path = format(n,'b').zfill(M)
    path = path[::-1]
    ans = S0
    t = S0
    for i in path:
        if(i=='1'):
            t*=d
        else:
            t*=u
        ans = min(ans, t)
    return ans


def look_bin(M=10,T=1,K=95,S0=100,r=0.04,sigma=0.25):
    dt = 1.0*T/M
    u = np.exp(sigma*np.sqrt(dt))
    d = np.exp(-sigma*np.sqrt(dt))
    p = (np.exp(r*dt) - d)/(u-d)
    V = []
    M = int(M)
    for i in range(M+1):
        A = np.zeros(int(pow(2,i)))
        V.append(A)
    
    for i in range(int(pow(2,M))):
        tp = comp_min(i,M,S0,u,d)
        V[M][i] =  max(0, K - tp)
    
    for i in range(M-1,-1,-1):
        for j in range(int(pow(2,i))):
            V[i][j] = np.exp(-r*dt)*(p*V[i+1][2*j] + (1-p)*V[i+1][2*j + 1])
    
    return V[0][0]

print(look_bin())