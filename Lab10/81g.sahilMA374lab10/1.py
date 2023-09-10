import numpy as np
import math
import matplotlib.pyplot as plt
import random

def GBM_paths(mu = 0.1, sigma = 0.2, S0 = 100,n=252):
    dt = 1.0/252
    i = 0
    W = []
    W.append(S0)
    while(i<n-1):
        i+=1
        w = np.random.normal()
        W.append(W[i-1]*math.exp(  (mu - (sigma**2)*0.5)*dt + sigma*math.sqrt(dt)*w ) )
    return W


def gen_paths(mu = 0.1, sigma = 0.2,S0 = 100, title="Asset Pricing",n=252):
    time = np.arange(n)
    for i in range(10):
        W = GBM_paths(mu,sigma,S0,n)
        plt.plot(time, W)
    
    plt.xlabel('time, t (in days)')
    plt.ylabel('Stock prices, S(t)')
    plt.title(title)
    plt.show()
    plt.close()

def asian_option_pricing(iter = 1000,r=0.05,sigma=0.2,S0=100,K=105,n = 252):
    asian_call =[]
    asian_put = []
    dt = 1/252
    for i in range(iter):
        S = GBM_paths(r,sigma,S0,n)
        call = max(np.mean(S) - K, 0)
        put = max(K-np.mean(S),0)
        asian_call.append(np.exp(-r*n*dt)*call)
        asian_put.append(np.exp(-r*n*dt)*put)
    return np.mean(asian_call),np.var(asian_call),np.mean(asian_put),np.var(asian_put)

##### Asset Pricing ####

gen_paths(0.1,0.2,100,"Asset Pricing: Real World")
gen_paths(0.05,0.2,100,"Asset Pricing: Risk-Neutral World")

##### Asian Options #####

K = [105,110,90]
for k in K:
    print(f'For K = {k}')
    call_mean,call_var,put_mean,put_var = asian_option_pricing(1000,0.05,0.2,100,k,126)
    print(f"Asian Call Option Price:\t\t{call_mean}")
    print(f"Asian Call Option Variance:\t\t{call_var}")
    print(f"Asian Put Option Price:\t\t\t{put_mean}")
    print(f"Asian Put Option Variance:\t\t{put_var}")
    print("\n\n")


#### Sensitivity Analysis ####

variation_with = ["Stock Price (S0)", "Strike Price (K)", "Volatility (sigma)", "Riske-free Rate (r)"]
A = []
S_range = [80,130]
K_range = [80,130]
sigma_range = [0.01,0.6]
r_range = [0.02,0.60]
tt = [S_range,K_range,sigma_range,r_range]
for i in range(4):
    tp = np.linspace(tt[i][0],tt[i][1],250)
    A.append(tp)

for i in range(4):
    asian_call, asian_put =[],[]
    r = 0.05
    S0 = 100
    sigma = 0.2
    K = 110
    for j in range(250):
        if(i == 0):
            S0 = A[i][j]
        elif(i==1):
            K = A[i][j]
        elif(i==2):
            sigma = A[i][j]
        else:
            r = A[i][j]
        call,_,put,_ = asian_option_pricing(100,r,sigma,S0,K,126)
        asian_call.append(call)
        asian_put.append(put)
    plt.subplot(1,2,1)
    plt.plot(A[i],asian_call)
    plt.xlabel(f"{variation_with[i]}")
    plt.ylabel("Asian Call Option Price")
    plt.title(f"Variation of Asian Call Option with {variation_with[i]}")

    plt.subplot(1,2,2)
    plt.plot(A[i],asian_put)
    plt.xlabel(f"{variation_with[i]}")
    plt.ylabel("Asian Put Option Price")
    plt.title(f"Variation of Asian Put Option with {variation_with[i]}")

    plt.show()
    plt.close()