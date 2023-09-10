import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import io

def compute(filename,tp):

    def wweights(M,C,mu):
        u = [1 for i in range(len(M))]
        tp1 = np.linalg.det([[1, u@np.linalg.inv(C)@np.transpose(M)],[mu, M@np.linalg.inv(C)@np.transpose(M)]])
        tp2 = np.linalg.det([[ u@np.linalg.inv(C)@np.transpose(u), 1],[ M@np.linalg.inv(C)@np.transpose(u), mu]])
        tp3 = np.linalg.det([[ u@np.linalg.inv(C)@np.transpose(u), u@np.linalg.inv(C)@np.transpose(M)],[ M@np.linalg.inv(C)@np.transpose(u), M@np.linalg.inv(C)@np.transpose(M)]])
        tp1/=tp3
        tp2/=tp3
        w = (tp1*(u@np.linalg.inv(C)) + tp2*(M@np.linalg.inv(C)))
        return w

    def compute_beta():
        beta = []
        df = pd.read_csv(filename)
        df.set_index('Date', inplace=True)
        df = df.pct_change()*len(df)/5
        sigma = df.std()
        sigma = sigma[0]
        stocks_name = list(df.columns)
        df1 = list(df[tp])
        df1.pop(0)
        for i in range(20):
            df2 = list(df[stocks_name[i+1]])
            df2.pop(0)
            cov = np.cov(df2, df1)
            beta.append(cov[0][1]/sigma**2)

        return beta


    df = pd.read_csv(filename)
    df.set_index('Date', inplace=True)
    df = df.pct_change()*len(df)/5
    M, sigma1, C = np.mean(df, axis = 0), df.std(), df.cov()
    stocks_name = list(df.columns)
    df = df.iloc[:,1:]
    M11, C11  =np.mean(df, axis = 0), df.cov()
    
    returns = np.linspace(-2, 5, num = 5000)
    u = np.array([1 for i in range(len(M11))])
    risk = []

    for mu in returns:
        w = wweights(M11, C11, mu)
        sigma = math.sqrt(w @ C11 @ np.transpose(w))
        risk.append(sigma)
    
    weight_min_var = u@np.linalg.inv(C11) / (u @ np.linalg.inv(C11) @ np.transpose(u))
    mu_min_var = weight_min_var@np.transpose(M11)
    risk_min_var = math.sqrt(weight_min_var @ C11 @ np.transpose(weight_min_var))
    returns_plot1, risk_plot1, returns_plot2, risk_plot2 = [], [], [], []
    for i in range(len(returns)):
        if returns[i] >= mu_min_var: 
            returns_plot1.append(returns[i])
            risk_plot1.append(risk[i])
        else:
            returns_plot2.append(returns[i])
            risk_plot2.append(risk[i])

    mu_rf = 0.05

    # market_portfolio_weights = (M - mu_rf * u) @ np.linalg.inv(C) / ((M - mu_rf * u) @ np.linalg.inv(C) @ np.transpose(u) )
    mu_market = M[0]
    risk_market = sigma1[0]

    plt.plot(risk_plot1, returns_plot1, color = 'Orange', label = 'Efficient frontier')
    plt.plot(risk_plot2, returns_plot2, color = 'Green')
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns") 
    plt.title("Minimum Variance Curve & Efficient Frontier")
    plt.plot(risk_market, mu_market, color = 'green', marker = 'o')
    plt.annotate('Market Portfolio (' + str(round(risk_market, 2)) + ', ' + str(round(mu_market, 2)) + ')', xy=(risk_market, mu_market), xytext=(0.2, 0.6))
    plt.plot(risk_min_var, mu_min_var, color = 'blue', marker = 'o')
    plt.annotate('Minimum Variance Portfolio (' + str(round(risk_min_var, 2)) + ', ' + str(round(mu_min_var, 2)) + ')', xy=(risk_min_var, mu_min_var), xytext=(risk_min_var, -0.6))
    plt.legend()
    plt.show()

    print("\npart b\n")
    print("Return = ", mu_market)
    print("Risk = ", risk_market)

    print("\npart c\n")
    returns_cml = []
    risk_cml = np.linspace(0, 2, num = 5000)
    for i in risk_cml:
        returns_cml.append(mu_rf+(mu_market - mu_rf)* i/ risk_market)


    slope, intercept = (mu_market - mu_rf) / risk_market, mu_rf
    print("\nEquation of CML is:")
    print("y = {:.2f} x + {:.2f}\n".format(slope, intercept))

    ###################################################################

    # plt.plot(risk, returns, color = 'Blue', label = 'Minimum Variance Curve')
    # plt.plot(risk_cml, returns_cml, color = 'Green', label = 'CML')
    # plt.title("Capital Market Line with Markowitz Efficient Frontier")
    # plt.xlabel("Risk (sigma)")
    # plt.ylabel("Returns") 
    # plt.legend()
    # plt.show()
    ##########################################################
    plt.plot(risk_cml, returns_cml)
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns") 
    plt.title("Capital Market Line")
    plt.legend(['CML'], loc="upper left")
    plt.show()
    ###################################################################

    betas = compute_beta()
    
    print("\n\nStocks Name\t\t\tActual Return\t\t\tExpected Return\t\t\tExpected Beta Value\n")
    bb1, bb2, m1, m2 =[], [] , [] , []
    for i in range(len(M11)):
        print(f"{stocks_name[i+1]:>12}\t\t\t{M11[i]:>12}\t\t{betas[i] * (mu_market - mu_rf) + mu_rf:>12}\t\t{betas[i]:>12}")
        if(i<10):
            bb1.append(betas[i])
            m1.append(M11[i])
        else:
            bb2.append(betas[i])
            m2.append(M11[i])

    beta_k = np.linspace(0.2, 1.5, 5000)
    mu_k = mu_rf + (mu_market - mu_rf)*beta_k
    plt.scatter(bb1, m1, color='blue',marker = 'o', label = 'Index Stocks')
    plt.scatter(bb2, m2, color='red',marker = 'o', label = 'Non Index Stocks')
    plt.plot(beta_k, mu_k)

    print("Equation of Security Market Line is:")
    print("mu = {:.2f} beta + {:.2f}".format(mu_market - mu_rf, mu_rf))

    plt.title('Security Market Line for all the 20 assets')
    plt.legend(['SML', 'Index Stocks', 'Non Index Stocks'],loc="upper left")
    plt.xlabel("Beta")
    plt.ylabel("Mean Return")
    plt.show()

print("The code is currently running for BSE (Sensex)\n")
compute('soham2.csv', 'NIFTY50')
# print("\n\n\nThe code is currently running for NSE (Nifty)\n")
# compute('nsedata1.csv', '^NSEI')
