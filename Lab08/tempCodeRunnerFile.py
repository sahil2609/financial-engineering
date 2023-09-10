std = []
        stocks = list(df.columns)
        stocks.pop(0)
        for i in stocks:
            x1 = np.log(df_reduced[i]/df_reduced[i].shift(1))
            s = np.nanstd(x1)
            std.append(s*np.sqrt(252))