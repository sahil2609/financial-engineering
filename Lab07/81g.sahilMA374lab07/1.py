import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits import mplot3d


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
t = [0,0.2,0.4,0.6,0.8,1]

C, P = calc_price(2, 1, 1, 0.05, 0.6,0.2)
print("Taking parameters: \ns=2\nT=1\nK=1\nr=0.05\nsigma=0.6\nt=0.2")
print("European Call Price =", C)
print("European Put Price =", P)