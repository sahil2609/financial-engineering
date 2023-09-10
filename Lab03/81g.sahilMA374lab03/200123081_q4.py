import math
import numpy as np
import time
from tabulate import tabulate
import matplotlib.pyplot as plt

S0 = 100
T = 1.0
r = 0.08
sigma = 0.20
K = 100

dp = {}

def binomial_model(M = 100):
    M = int(M)
    # print(M)
    # print(T)
    # print((T*1.0)/(M*1.0))
    dt = (T*1.0)/(M*1.0)
    dim = M+ 1
    u1 = math.exp(sigma*(dt**0.5))
    d1 = math.exp(-1*sigma*(dt**0.5))

    u2 = math.exp(sigma*(dt**0.5) + (r - 0.5*(sigma**2))*dt)
    d2 = math.exp(-sigma*(dt**0.5) + (r - 0.5*(sigma**2))*dt) 
    # print(u1)
    # print(d1)
    #risk neutral probability
    q1 = (math.exp(r*dt) - d1)/(u1 - d1)
    q2 = (math.exp(r*dt) - d2)/(u2 - d2)

    factors1 = np.zeros((dim, dim))
    factors1[0][0] = 1

    factors2 = np.zeros((dim, dim))
    factors2[0][0] = 1


    call1= np.zeros((dim, dim))
    put1 = np.zeros((dim, dim))
    call2 = np.zeros((dim, dim))
    put2 = np.zeros((dim, dim))

    for i in range(1,dim):
        for j in range(i):
            factors1[i][j] = factors1[i-1][j]*u1
            factors1[i][j+1] = factors1[i-1][j]*d1
            factors2[i][j] = factors2[i-1][j]*u2
            factors2[i][j+1] = factors2[i-1][j]*d2

    for i in range(dim):
        call2[dim - 1][i] = max(0, factors2[dim -1][i]*S0 - K)

    for i in range(dim -2, -1,-1):
        for j in range(i+1):
            call2[i][j] = math.exp(-r*dt)*(q2*call2[i+1][j] + (1 - q2)*(call2[i+1][j+1]))

    return call2[0][0]


def markokv_helper(M,dt,q,u,d,Sm,ct,n):
    if n == M:
        return max(Sm - K, 0)
    if dp.get((n,ct))!= None:
        return dp[n,ct]
    up = markokv_helper(M,dt,q,u,d,Sm*u,ct+1,n+1)
    dn = markokv_helper(M,dt,q,u,d,Sm*d,ct,n+1)
    dp[n,ct] = math.exp(-r*dt)*(q*up + (1-q)*dn)
    return dp[n,ct]

def markov_algo(M):
    delt = (T*1.0)/(M*1.0)
    u = math.exp(sigma*(delt**0.5) + (r - 0.5*(sigma**2))*delt)
    d = math.exp(-sigma*(delt**0.5) + (r - 0.5*(sigma**2))*delt) 
    M = int(M)
    q = (math.exp(r*delt) - d)/(u - d)
    dp.clear()
    price = markokv_helper(M,delt,q,u,d,S0,0,0)
    return price


M = [5,10,15,20,25,50,100]
print("Using Markov based Computation for European Call")
for i in M:
    s = round(time.time(), 10)
    value = markov_algo(i)
    e = round(time.time(), 10)
    print(f'Option Value for M = {i} is {value} and time taken is {e-s}')

M1 = [5,10,20]
print("Using Binomial Model for European Call")
for i in M1:
    s = round(time.time(), 10)
    value = binomial_model(i)
    e = round(time.time(), 10)
    print(f'Option Value for M = {i} is {value} and time taken is {e-s}')