import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

T = 1
def binomial_pricing_model(S0 = 100, K = 100,  r = 8, sigma = 20, M = 100):
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
        call1[dim - 1][i] = max(0, factors1[dim -1][i]*S0 - K)
        call2[dim - 1][i] = max(0, factors2[dim -1][i]*S0 - K)
        put1[dim - 1][i] = max(0, K - factors1[dim -1][i]*S0)
        put2[dim - 1][i] = max(0, K - factors2[dim -1][i]*S0)

    for i in range(dim -2, -1,-1):
        for j in range(i+1):
            call1[i][j] = math.exp(-r*dt)*(q1*call1[i+1][j] + (1 - q1)*(call1[i+1][j+1]))
            call2[i][j] = math.exp(-r*dt)*(q2*call2[i+1][j] + (1 - q2)*(call2[i+1][j+1]))
            put1[i][j] = math.exp(-r*dt)*(q1*put1[i+1][j] + (1 - q1)*(put1[i+1][j+1]))
            put2[i][j] = math.exp(-r*dt)*(q2*put2[i+1][j] + (1 - q2)*(put2[i+1][j+1]))

    return call1[0][0], put1[0][0], call2[0][0], put2[0][0]


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

call2D = np.zeros((7,2,N))
put2D = np.zeros((7,2,N))
call3D = np.zeros((5,5,2,N,N))
put3D = np.zeros((5,5,2,N,N))

l_limit = [50, 50, 0.06, 0.05, 50,50,50]
r_limit = [150, 150, 0.7, 0.35, 150,150,150]

for i in range(7):
    A[i] = np.linspace(l_limit[i], r_limit[i], N)

for i in range(N):
    call2D[0][0][i], put2D[0][0][i], call2D[0][1][i], put2D[0][1][i] = binomial_pricing_model(A[0][i], K, r, sigma, M)
    call2D[1][0][i], put2D[1][0][i], call2D[1][1][i], put2D[1][1][i] = binomial_pricing_model(S0, A[1][i], r, sigma, M)
    call2D[2][0][i], put2D[2][0][i], call2D[2][1][i], put2D[0][1][i] = binomial_pricing_model(S0, K, A[2][i], sigma, M)
    call2D[3][0][i], put2D[3][0][i], call2D[3][1][i], put2D[3][1][i] = binomial_pricing_model(S0, K, r, A[3][i], M)
    call2D[4][0][i], put2D[4][0][i], call2D[4][1][i], put2D[4][1][i] = binomial_pricing_model(S0, 95, r, sigma, A[4][i])
    call2D[5][0][i], put2D[5][0][i], call2D[5][1][i], put2D[5][1][i] = binomial_pricing_model(S0, 100, r, sigma, A[4][i])
    call2D[6][0][i], put2D[6][0][i], call2D[6][1][i], put2D[6][1][i] = binomial_pricing_model(S0, 105, r, sigma, A[4][i])

for i in range(N):
    for j in range(N):
        call3D[0][1][0][i][j], put3D[0][1][0][i][j], call3D[0][1][1][i][j], put3D[0][1][1][i][j] = binomial_pricing_model(A[0][i], A[1][j], r, sigma, M)
        call3D[0][2][0][i][j], put3D[0][2][0][i][j], call3D[0][2][1][i][j], put3D[0][2][1][i][j] = binomial_pricing_model(A[0][i], K, A[2][j], sigma, M)
        call3D[0][3][0][i][j], put3D[0][3][0][i][j], call3D[0][3][1][i][j], put3D[0][3][1][i][j] = binomial_pricing_model(A[0][i], K, r, A[3][j], M)
        call3D[0][4][0][i][j], put3D[0][4][0][i][j], call3D[0][4][1][i][j], put3D[0][4][1][i][j] = binomial_pricing_model(A[0][i], K, r, sigma, A[4][j])
        call3D[1][2][0][i][j], put3D[1][2][0][i][j], call3D[1][2][1][i][j], put3D[1][2][1][i][j] = binomial_pricing_model(S0, A[1][i], A[2][j], sigma, M)
        call3D[1][3][0][i][j], put3D[1][3][0][i][j], call3D[1][3][1][i][j], put3D[1][3][1][i][j] = binomial_pricing_model(S0, A[1][i], r, A[3][j], M)
        call3D[1][4][0][i][j], put3D[1][4][0][i][j], call3D[1][4][1][i][j], put3D[1][4][1][i][j] = binomial_pricing_model(S0, A[1][i], r, sigma, A[4][j])
        call3D[2][3][0][i][j], put3D[2][3][0][i][j], call3D[2][3][1][i][j], put3D[2][3][1][i][j] = binomial_pricing_model(S0, K, A[2][i], A[3][j], M)
        call3D[2][4][0][i][j], put3D[2][4][0][i][j], call3D[2][4][1][i][j], put3D[2][4][1][i][j] = binomial_pricing_model(S0, K, A[2][i], sigma, A[4][j])
        call3D[3][4][0][i][j], put3D[3][4][0][i][j], call3D[3][4][1][i][j], put3D[3][4][1][i][j] = binomial_pricing_model(S0, K, r, A[3][i], A[4][j])

x_label = ['Initial Stock Price', 'Strike Price', 'Risk Free Rate', 'Volatility', 'No of steps','No of steps','No of steps']
for i in range(7):
    fig1, ax1 = plt.subplots(2,2)
    ax1[0,0].plot(A[i], call2D[i][0], label='Call Prices')
    ax1[0,1].plot(A[i], put2D[i][0], label='Put Prices')
    ax1[1,0].plot(A[i], call2D[i][1], label='Call Prices')
    ax1[1,1].plot(A[i], put2D[i][1], label='Put Prices')
    ax1[0,0].legend()
    ax1[0,1].legend()
    ax1[1,0].legend()
    ax1[1,1].legend()
    ax1[0,0].set_title(f'Call Option Price vs {x_label[i]} (Set 1 of u,d)')
    ax1[0,1].set_title(f'Put Option Price vs {x_label[i]} (Set 1 of u,d)')
    ax1[1,0].set_title(f'Call Option Price vs {x_label[i]} (Set 2 of u,d)')
    ax1[1,1].set_title(f'Put Option Price vs {x_label[i]} (Set 2 of u,d)')
    ax1[0,0].set_xlabel(f'{x_label[i]}')
    ax1[0,1].set_xlabel(f'{x_label[i]}')
    ax1[1,0].set_xlabel(f'{x_label[i]}')
    ax1[1,1].set_xlabel(f'{x_label[i]}')
    ax1[0,0].set_ylabel('Option Price')
    ax1[0,1].set_ylabel('Option Price')
    ax1[1,0].set_ylabel('Option Price')
    ax1[1,1].set_ylabel('Option Price')

    plt.tight_layout()
    plt.show()

for i in range(5):
    for j in range(i + 1, 5):
        fig1 = plt.figure(figsize=(12, 12))
        fig2 = plt.figure(figsize=(12, 12))
        ax1 = fig1.add_subplot(121, projection='3d')
        ax2 = fig1.add_subplot(122, projection='3d')
        ax3 = fig2.add_subplot(121, projection='3d')
        ax4 = fig2.add_subplot(122, projection='3d')

        X = np.zeros(N * N)
        Y = np.zeros(N * N)
        for ii in range(N):
            for jj in range(N):
                X[N * ii + jj] = A[i][ii]
                Y[N * ii + jj] = A[j][jj]
        ax1.scatter(X, Y, call3D[i][j][0].ravel(), label='Call Prices')
        ax2.scatter(X, Y, put3D[i][j][0].ravel(), label='Put Prices')
        ax3.scatter(X, Y, call3D[i][j][1].ravel(), label='Call Prices')
        ax4.scatter(X, Y, put3D[i][j][1].ravel(), label='Put Prices')

        ax1.set_title(f'Call Price vs ({x_label[i]} and {x_label[j]}) (Set 1 of u,d)')
        ax1.set_xlabel(f'{x_label[i]}')
        ax1.set_ylabel(f'{x_label[j]}')
        ax1.set_zlabel('Call Price')

        ax2.set_title(f'Put Price vs ({x_label[i]} and {x_label[j]}) (Set 1 of u,d)')
        ax2.set_xlabel(f'{x_label[i]}')
        ax2.set_ylabel(f'{x_label[j]}')
        ax2.set_zlabel('Put Price')

        ax3.set_title(f'Call Price vs ({x_label[i]} and {x_label[j]}) (Set 2 of u,d)')
        ax3.set_xlabel(f'{x_label[i]}')
        ax3.set_ylabel(f'{x_label[j]}')
        ax3.set_zlabel('Call Price')

        ax4.set_title(f'Put Price vs ({x_label[i]} and {x_label[j]}) (Set 2 of u,d)')
        ax4.set_xlabel(f'{x_label[i]}')
        ax4.set_ylabel(f'{x_label[j]}')
        ax4.set_zlabel('Put Price')

        plt.tight_layout()
        plt.show()