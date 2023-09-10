import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import norm


def plot3d(x,y,z):
    ax = plt.axes(projection = '3d')
    

def N(x):
    return norm.cdf(x)

def cal_prices(S,K,t,T,r,sigma):
    if(T == t):
        return max(0, S- K), max(0, K-S)
    t = T - t
    d1 = (np.log(S/K) + (r + (sigma**2)/2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    C = S*N(d1)  -  K*np.exp(-r*t)*N(d2)
    P = -S*N(-d1) + K*np.exp(-r*t)*N(-d2)
    return C,P














# def comp_min(n,M,S0,u,d):
#     path = format(n,'b').zfill(M)
#     path = path[::-1]
#     ans = S0
#     t = S0
#     for i in path:
#         if(i=='1'):
#             t*=d
#         else:
#             t*=u
#         ans = min(ans, t)
#     return ans


# def look_bin(M=10,T=1,K=95,S0=100,r=0.04,sigma=0.25):
#     dt = 1.0*T/M
#     u = np.exp(sigma*np.sqrt(dt))
#     d = np.exp(-sigma*np.sqrt(dt))
#     p = (np.exp(r*dt) - d)/(u-d)
#     V = []
#     M = int(M)
#     for i in range(M+1):
#         A = np.zeros(int(pow(2,i)))
#         V.append(A)
    
#     for i in range(int(pow(2,M))):
#         tp = comp_min(i,M,S0,u,d)
#         V[M][i] =  max(0, K - tp)
    
#     for i in range(M-1,-1,-1):
#         for j in range(int(pow(2,i))):
#             V[i][j] = np.exp(-r*dt)*(p*V[i+1][2*j] + (1-p)*V[i+1][2*j + 1])
    
#     return V[0][0]

# print(look_bin())



# df1 = pd.read_csv("bsedata1")
# df1.set_index("Date", inplace=True)

# dp ={}
# #markov
# def markov(m,n,dt,s0,ms0,q,u,d):
#     if(m==n):
#         return ms0-s0
#     if(dp.get((s0, ms0))!= None):
#         return dp[s0,ms0]
#     up = markov(m, n+1, s0*u, max(s0*u, ms0), q,u,d)
#     dn = markov(m, n+1, s0*d, max(s0*d, ms0), q,u,d)

#     dp[s0, ms0] = math.sqrt(-r*dt)*(q*up + (1-q)*dn)
#     return dp[s0,ms0]


# #minimum variance
# def min_var(m,c):
#     u = np.ones(len(m))
#     w = (u@np.linalg.inv(c))/(u@np.linalg.inv(c)@np.transpose(u))
#     mu = m@np.transpose(w)
#     sigma = np.sqrt(w@c@np.transpose(w))
#     return mu, sigma

# #weights with given mu
# def get_weights(m,c,mu):
#     u = np.ones((len(m)))
#     t1 = np.linalg.det([[1, u@np.linalg.inv(c)@np.transpose(m)],[mu, m@np.linalg.inv(c)@np.transpose(m)]])
#     t2 = np.linalg.det([u@np.linalg.inv(c)@np.transpose(u),1],[m@np.linalg.inv(c)@np.transpose(m),mu])
#     t3 = np.linalg.det([u@np.linalg.inv(c)@np.transpose(u), u@np.linalg.inv(c)@np.transpose(m)],[m@np.linalg.inv(c)@np.transpose(u), m@np.linalg.inv(c)@np.transpose(m)])
#     w = (t1*(mu@np.linalg.inv(c)) + t2*(m@np.linalg.inv(c)))/(t3)
#     muu = m@np.transpose(w)
#     sigma = np.sqrt(w@c@np.transpose(w))
#     return muu, sigma

# #binomial_model
# def binomial_model(S0, T, M, K, r, sigma, u, d):
#     dt = 1.0*T/M
#     p  = (math.exp(r*dt) - d)/(u-d)
#     call, put, factors = [],[],[]
#     for i in range(M+1):
#         j = i + 1
#         t1 = np.zeros((j))
#         t2 = np.zeros((j))
#         t3 = np.zeros((j))
#         call.append(t1)
#         put.append(t2)
#         factors.append(t3)
    
#     factors[0][0] = 1
#     for i in range(1, M+1):
#         for j  in range(i):
#             factors[i][j] = factors[i-1][j]*u
#             factors[i][j+1] = factors[i-1][j]*d
    
#     for i in range(M+1):
#         call[M][i] = max(0, S0*factors[M][i] - K)
#         put[M][i] = max(0, K - S0*factors[M][i])
    
#     for i in range(M-1, -1,-1):
#         for j in range(i+1):
#             call[i][j] = np.exp(-r*dt)*(p*call[i+1][j]  + (1-p)*call[i+1][j+1])
#             put[i][j] = np.exp(-r*dt)*(p*put[i+1][j]  + (1-p)*put[i+1][j+1])
#     print(call[0][0])
#     print(put[0][0])


# X = np.linspace(1,2,1000)
# Y = np.linspace(8,11,1000)
# Z = np.linspace(10,30,1000)
# fig =  plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.scatter(X,Y,Z, label ="tp")
# ax.set_xlabel("fsdsfd")

# plt.legend()
# plt.show()


# M = 10
# i = 5
# path = format(i, 'b').zfill(M)
# path = path[::-1]
# print(path)

# S0 = 100
# K = 100
# r = 0.08
# sigma=0.20
# M = 100
# T = 1
# dt = 1.0*T/M
# u2 = math.exp(sigma*(dt**0.5) + (r - 0.5*(sigma**2))*dt)
# d2 = math.exp(-sigma*(dt**0.5) + (r - 0.5*(sigma**2))*dt) 
# binomial_model(S0,T,M,K,r,sigma,u2,d2)

