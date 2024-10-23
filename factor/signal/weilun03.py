from factor.factors import Factor
import pandas as pd
import numpy as np



class weilun03(Factor):
    need = ["close"]
    def __init__(self, n:int=-1):
        self.n = n
        self.mid = self.n/2
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["close"]
        returns = np.log(price.iloc[-1]) - np.log(price.iloc[-self.n])
        # print(x)
        coin_close = x[x.columns[x.columns.str.contains("close")]]
        # print(coin_close)
        coin_returns = coin_close.iloc[-1].apply(np.log) - coin_close.iloc[-self.n].apply(np.log)
        # print(coin_returns)
        # exit()
        return sum(returns > coin_returns) / len(coin_returns)
    
    def __str__(self) -> str:
        return f"group return rank {self.n}"