import numpy as np
import math
import matplotlib.pyplot  as plt


def CIR(beta, mu, sigma,r,t,T):
    a = beta
    b = mu
    gamma = np.sqrt(a**2 + 2*sigma**2)
    B = (2*(np.exp(gamma*(T-t)) -1 ))/((gamma + a)*(np.exp(gamma*(T-t)) -1 ) + 2*gamma)
    A = np.power((2*gamma*np.exp((a + gamma)*(0.5*(T-t))))/((gamma + a)*(np.exp(gamma*(T-t)) -1 ) + 2*gamma),(2*a*b)/(sigma**2))
    P = A*np.exp(- B*r)
    y = -math.log(P) / (T - t)
    return y

values = [[0.02, 0.7, 0.02, 0.1], [0.7, 0.1, 0.3, 0.2], [0.06, 0.09, 0.5, 0.02]]

T = np.linspace(0.01, 10, num=10)
for i in range(len(values)):
    Y = []
    for j in range(10):
        y = CIR(values[i][0],values[i][1],values[i][2],values[i][3],0,T[j])
        Y.append(y)
    plt.plot(T, Y, marker='o')
    plt.xlabel('Maturity (T)')
    plt.ylabel('Yield')
    plt.title(f'Term structure for parameter set - {i+1}')
    plt.show()

T = np.linspace(0.1, 600, num=600)
r_list = [0.1 * i for i in range(1, 11)]
for idx in range(1):
    beta, mu, sigma, r = values[idx]
    for r in r_list:
        Y = []
        for k in T:
            r = np.round(r,2)
            y = CIR(beta, mu, sigma, r, 0, k)
            Y.append(y)
        plt.plot(T, Y, label = f'r = {r}')

    plt.xlabel('Maturity (T)')
    plt.ylabel('Yield')
    plt.title(f'Term structure for 10 different values of r(0) with parameter set - {idx+1}')
    plt.legend(loc = 'upper right')
    plt.show()
