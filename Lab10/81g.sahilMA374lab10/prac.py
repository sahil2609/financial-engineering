import numpy as np
import matplotlib.pyplot as plt
import math


def GBM(n,S0,sigma,mu):
    S = []
    S.append(S0)
    S1 = []
    SS = S0
    S1.append(SS)
    dt = 1.0/252
    for i in range(n-1):
        u = np.random.normal()
        u1 = -u
        S.append(S0*np.exp((mu - (sigma**2)/2)*dt + (sigma*np.sqrt(dt)*u)))
        S0 = S[len(S)-1]
        S1.append(SS*np.exp((mu - (sigma**2)/2)*dt + (sigma*np.sqrt(dt)*u1)))
        SS = S1[len(S1)-1]
    return S,S1



def asian_pricing(iter,n,S0,sigma,mu,K):
    dt = 1.0/252
    P1 = []
    P2 = []
    for i in range(iter):
        S,S1 = GBM(n,S0,sigma,mu)
        p1 = np.mean(S)
        p2 = np.mean(S)
        p1 = (np.exp(-mu*n*dt)*max(p1-K,0))
        p2 = (np.exp(-mu*n*dt)*max(p2-K,0))
        P1.append(0.5*(p1 + p2))
    return np.mean(P1), np.var(P1)

print(asian_pricing(1000,126,100,0.2,0.05,105))