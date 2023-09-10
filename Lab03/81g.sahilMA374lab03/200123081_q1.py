import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

T = 1
def binomial_pricing_model(S0 = 100, K = 100,  r = 0.08, sigma = 0.20, M = 100):
    M = int(M)
    # print(M)
    # print(T)
    # print((T*1.0)/(M*1.0))
    dt = (T*1.0)/(M*1.0)
    dim = M+ 1
    u1 = math.exp(sigma*(dt**0.5) + (r - 0.5*(sigma**2))*dt)
    d1 = math.exp(-sigma*(dt**0.5) + (r - 0.5*(sigma**2))*dt) 
    # print(u1)
    # print(d1)
    #risk neutral probability
    q1 = (math.exp(r*dt) - d1)/(u1 - d1)

    factors1 = np.zeros((dim, dim))
    factors1[0][0] = 1


    call1= np.zeros((dim))
    put1 = np.zeros((dim))

    curr = S0*(u1**M)
    for i in range(dim):
        call1[i] = max(0 , curr - K)
        put1[i] = max(0, K - curr)
        curr *= d1/u1

    for i in range(M):
        for j in range(M - i):
            p_put = max(0, - S0*(u1**(M - i - j -1))*(d1**(j)) + K)
            p_call = max(0, + S0*(u1**(M - i - j -1))*(d1**(j)) - K)
            put1[j] = max(p_put, math.exp(-r*dt)*(q1*put1[j] + (1-q1)*put1[j+1]))
            call1[j] = max(p_call, math.exp(-r*dt)*(q1*call1[j] + (1-q1)*call1[j+1]))
    return call1[0], put1[0]


#initialization
S0 = 100
K = 100
T = 1
M = 100
r = 0.08
sigma = 0.2
dt = T/M

#sensitivity analysis
N = 30
A = np.zeros((7, N))
B = np.zeros((3, 100))


call2D = np.zeros((7,2,N))
put2D = np.zeros((7,2,N))

call22 = np.zeros((3,2,100))
put22 = np.zeros((3,2,100))


l_limit = [50, 50, 0.06, 0.05, 50,50,50]
r_limit = [150, 150, 0.7, 0.35, 150,150,150]

for i in range(4):
    A[i] = np.linspace(l_limit[i], r_limit[i], N)

B[0] = np.linspace(50, 150, 100)
B[1] = np.linspace(50, 150, 100)
B[2] = np.linspace(50, 150, 100)


tp1 , tp2 = binomial_pricing_model()
print(f'Call option price = {tp1} and Put option price = {tp2}')

for i in range(N):
    call2D[0][0][i], put2D[0][0][i] = binomial_pricing_model(A[0][i], K, r, sigma, M)
    call2D[1][0][i], put2D[1][0][i] = binomial_pricing_model(S0, A[1][i], r, sigma, M)
    call2D[2][0][i], put2D[2][0][i] = binomial_pricing_model(S0, K, A[2][i], sigma, M)
    call2D[3][0][i], put2D[3][0][i] = binomial_pricing_model(S0, K, r, A[3][i], M)

for i in range(100):
    
    call22[0][0][i], put22[0][0][i] = binomial_pricing_model(S0, 95, r, sigma, B[0][i])
    call22[1][0][i], put22[1][0][i] = binomial_pricing_model(S0, 100, r, sigma, B[1][i])
    call22[2][0][i], put22[2][0][i] = binomial_pricing_model(S0, 105, r, sigma, B[2][i])


x_label = ['Initial Stock Price', 'Strike Price', 'Risk Free Rate', 'Volatility', 'No of steps with K = 95','No of steps with K = 100','No of steps with K = 105']
for i in range(4):
    fig1, ax1 = plt.subplots(2)
    ax1[0].plot(A[i], call2D[i][0], label='Call Prices')
    ax1[1].plot(A[i], put2D[i][0], label='Put Prices')
    ax1[0].legend()
    ax1[1].legend()
    ax1[0].set_title(f'Call Option Price vs {x_label[i]}')
    ax1[1].set_title(f'Put Option Price vs {x_label[i]}')
    ax1[0].set_xlabel(f'{x_label[i]}')
    ax1[1].set_xlabel(f'{x_label[i]}')
    ax1[0].set_ylabel('Option Price')
    ax1[1].set_ylabel('Option Price')

    plt.tight_layout()
    plt.show()

for i in range(3):
    fig1, ax1 = plt.subplots(2)
    ax1[0].plot(B[i], call22[i][0], label='Call Prices')
    ax1[1].plot(B[i], put22[i][0], label='Put Prices')
    ax1[0].legend()
    ax1[1].legend()
    ax1[0].set_title(f'Call Option Price vs {x_label[4+i]}')
    ax1[1].set_title(f'Put Option Price vs {x_label[i+4]}')
    ax1[0].set_xlabel(f'{x_label[i+4]}')
    ax1[1].set_xlabel(f'{x_label[i+4]}')
    ax1[0].set_ylabel('Option Price')
    ax1[1].set_ylabel('Option Price')

    plt.tight_layout()
    plt.show()