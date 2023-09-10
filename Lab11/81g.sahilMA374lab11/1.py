import numpy as np
import math
import matplotlib.pyplot  as plt


def vasicek(beta, mu, sigma,r,t,T):
    a = beta
    b = beta*mu
    B = (1 - np.exp(-a*(T - t)))/a
    A = ((B - T + t)*(a*b - 0.5*sigma**2))/a**2 - (sigma**2 * B**2)/(4*a)
    P = np.exp(A - B*r)
    y = -math.log(P) / (T - t)
    return y

values = [[5.9, 0.2, 0.3, 0.1], [3.9, 0.1, 0.3, 0.2], [0.1, 0.4, 0.11, 0.1]]
T = np.linspace(0.01, 10, num=10)
for i in range(len(values)):
    Y = []
    for j in range(10):
        y = vasicek(values[i][0],values[i][1],values[i][2],values[i][3],0,T[j])
        Y.append(y)
    plt.plot(T, Y, marker='o')
    plt.xlabel('Maturity (T)')
    plt.ylabel('Yield')
    plt.title(f'Term structure for parameter set - {i+1}')
    plt.show()

T = np.linspace(10, 500, num=500)
r_list = [0.1 * i for i in range(1, 11)]
for idx in range(len(values)):
    beta, mu, sigma, r = values[idx]
    for r in r_list:
        Y = []
        for k in T:
            r = np.round(r,2)
            y = vasicek(beta, mu, sigma, r, 0, k)
            Y.append(y)
        plt.plot(T, Y, label = f'r = {r}')

    plt.xlabel('Maturity (T)')
    plt.ylabel('Yield')
    plt.title(f'Term structure for 10 different values of r(0) with parameter set - {idx+1}')
    plt.legend()
    plt.show()
