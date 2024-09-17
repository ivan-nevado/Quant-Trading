import backtesting as bt
import pandas as pd
import talib as ta
from backtesting import Strategy
from backtesting import Backtest
from backtesting.lib import crossover
import pandas_ta as tal

factor = 1000 # adjust this factor nased on price magnitudes
#factor allows you to trade with smaller amount like .1btc

class Gap(Strategy):
    stochrsi_smooth1 = 3
    stochrsi_smooth3 = 3
    bbands_length = 20
    stochrsi_length = 14
    bbands_std = 2

    def init(self):
        self.ema_period = 9 # period for exponential moving averages
        self.sma_period = 21 #period for simple moving average
        self.ema = self.I(ta.EMA, self.data.Close, self.ema_period)
        self.sma = self.I(ta.SMA, self.data.Close, self.sma_period)
        self.stock_rsi_k = self.I(stock_rsi_k, self.data)
        self.stoch_rsi_d = self.I(stoch_rsi_d, self.data)

    def next(self):
        price = self.data.Close[-1] #this is the current price

        #closing the order
        if self.position:
            if self.position.is_long:
                if self.data.Close[-1] > self.data.Close[-2]:
                    self.sell()
            elif self.position.is_short:
                if self.data.Close[-1] < self.data.Close[-2]:
                    self.buy()

        #if there is no position open
        if not self.position:
            if self.data.Open > self.data.Close[-1]: #gap up
                if crossover(self.ema, self.sma) and crossover(self.stock_rsi_k, self.stoch_rsi_d):
                    self.buy(size = .15, tp = price * 1.2, sl = price * .93)
                
            elif self.data.Open < self.data.Close[-1]: #price gaps down
                if crossover(self.sma, self.ema) and crossover(self.stock_rsi_k, self.stoch_rsi_d):
                    self.sell(size = .15, tp = price * .80, sl = price *1.09)


def bands(data):
    bbands = tal.bbands(close = data.Close.s, length = 20, std = 20)
    return bbands.to_numpy().T

def stock_rsi_k(data):
    stochrsi = tal.stochrsi(close= data.Close.s, k=3,d=3)
    return stochrsi['STOCHRSIk_14_14_3_3'].to_numpy()

def stoch_rsi_d(data):
    stochrsi = tal.stochrsi(close = data.Close.s, k=3,d=3)
    return stochrsi['STOCHRSId_14_14_3_3'].to_numpy()

data = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\Day 4\Algo_Orders\BTC-15m-100wks-data.csv')
data.open /= factor
data.high /= factor
data.low /= factor
data.close /= factor
data.volume *= factor

data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace = True)

bt = Backtest(data,Gap, cash = 10000)
result = bt.run()

bt.plot()

print(result)

