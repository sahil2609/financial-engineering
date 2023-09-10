import numpy as np
import math
import matplotlib.pyplot as plt
import random

def GBM_paths(mu = 0.1, sigma = 0.2, r = 0.05, S0 = 100 ):
    dt = 1.0/252
    i = 0
    W = []
    W.append(S0)
    while(i<251):
        i+=1
        w = np.random.normal()
        W.append(W[i-1]*math.exp(  (mu - sigma**2)*0.5*dt + sigma*math.sqrt(dt)*w ) )
    return W


def gen_paths(mu = 0.1, sigma = 0.2, r = 0.05, S0 = 100):
    time = np.arange(252)
    print(time)
    for i in range(10):
        W = GBM_paths(mu,sigma,r,S0)
        plt.plot(time, W)
    
    plt.xlabel('time, t (in days)')
    plt.ylabel('Stock prices, S(t)')
    plt.title("Asset Pricing: Real World")
    plt.show()
    plt.close()


gen_paths()
    