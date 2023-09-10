import math
import numpy as np
import time
from tabulate import tabulate

S0 = 100
T = 1.0
r = 0.08
sigma = 0.20
tme = []

def look_back(M,f=0):
    delt = T/(M*1.0)
    u = math.exp(sigma*(delt**0.5) + (r - 0.5*(sigma**2))*delt)
    d = math.exp(-sigma*(delt**0.5) + (r - 0.5*(sigma**2))*delt) 
    M = int(M)
    q = (math.exp(r*delt) - d)/(u - d)

    S = []
    V = []
    mx1 = [S0]
    for i in range(M + 1):
        tp = np.zeros(int(math.pow(2,i)))
        tp1 = np.zeros(int(math.pow(2,i)))
        S.append(tp)
        V.append(tp1)

    S[0][0] = S0
    for i in range(1, M+1):
        for j in range(int(math.pow(2,i-1))):
            S[i][2*j] = S[i-1][j]*u
            S[i][2*j + 1] = S[i-1][j]*d

        
    
        mx2 = np.zeros(int(math.pow(2, i)))
        for j in range(int(math.pow(2,i-1))):
            mx2[2*j] = max(mx1[int(j)], S[i][2*j])
            mx2[2*j +1] = max(mx1[int(j)], S[i][2*j + 1])
        mx1 = mx2
    
    
    for i in range(int(math.pow(2,M))):
        V[M][i] = mx1[i] - S[M][i]
    
    for i in range(M-1, -1,-1):
        for j in range(int(math.pow(2, i))):
            if j< (1<<i):
                V[i][j] = math.exp(-r*delt)*(q*V[i+1][2*j] + (1-q)*V[i+1][2*j+ 1])
            else:
                V[i][j] = ' '

    if f:
        # tab = np.zeros((32,6))
        # for i in range(M+1):
        #     for j in range(int(math.pow(2,i))):
        #         if V[i][j] != ' ':
        #             tab[j][i] = V[i][j]
        #         else:
        #             tab[j][i] = ' '

        
        print(tabulate(V,tablefmt="grid"))

    return V[0][0]


M = [5,10]
for i in M:
    s = round(time.time(), 10)
    value = look_back(i)
    e = round(time.time(), 10)
    tme.append(e -s)
    print(f'Initial price for M = {i} is {value}')

head = ["M", "Option Value"]
rows = []
for i in range(5,16):
    value = look_back(i)
    rows.append([i, value])

print(tabulate(rows,headers=head,tablefmt="grid"))

look_back(5,1)

