import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

r = 0.05
def fn(delay,s,k,p,sigma):
    d1 = (np.log(s/k) + (r + (sigma**2)/2)*delay)/(sigma*np.sqrt(delay))
    d2 = d1 - sigma*(delay**0.5)
    num = s*norm.cdf(d1) - k*np.exp(-r*delay)*norm.cdf(d2) - p
    den = s*norm.pdf(d1)*(delay**0.5)
    return [num, den]


def newton_raphson(delay,s,k,p):
    sig = 0.3
    i = 0
    while(i<1000):
        x = fn(delay,s,k,p,sig)
        if(x[1] == 0):
            return -1
        tp = sig - (x[0]/x[1])
        if(abs(tp - sig) <= 1e-6):
            return tp
        sig = tp
    return sig