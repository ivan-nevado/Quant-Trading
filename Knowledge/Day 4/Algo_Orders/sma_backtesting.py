import numpy as np
import pandas as pd
import talib 
from backtesting import Backtest, Strategy
# from backtesting.test import GOOG

class SMAPullbackStrategy(Strategy):
    stop_loss = 0
    n1 = 20 #20 sma

    def init(self):
        price = self.data.Close
        self.sma = self.I(talib.SMA, price, self.n1)

    def next(self):
        price = self.data.Close[-1]
        low_price_prev_day = self.data.Low[-1]

        if price > self.sma[-1]:
            self.stop_loss = low_price_prev_day

            if price <= 2.4 * self.stop_loss:
                self.buy()

#test the strategy and pull in the data
data = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\Day 4\Algo_Orders\BTC-USD (1).csv')

data.columns = [column.capitalize() for column in data.columns]

bt = Backtest(data, SMAPullbackStrategy, cash=1000000, commission=.002)
output = bt.optimize(maximize = 'Equity Final [$]', n1 = range(5,40,1))
print(output)

bt.plot()
        
