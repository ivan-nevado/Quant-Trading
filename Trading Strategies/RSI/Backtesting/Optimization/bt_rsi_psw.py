import pandas as pd
from backtesting import Backtest, Strategy
from pyswarm import pso

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Iván Quant Trading\TFG\BTC-6h-300wks-data.csv')

# Ensure 'Date' column exists and is properly formatted
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Function to calculate RSI (handling the backtesting.py internal data type)
def RSI(arr, period):
    series = pd.Series(arr)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Strategy Class
class EMARSI_Strategy(Strategy):
    ema_period = 150
    rsi_period = 10
    overbought = 70
    oversold = 30

    def init(self):
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, min_periods=self.ema_period).mean(), self.data.Close)
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

    def next(self):
        ema_value = self.ema[-1]
        rsi_value = self.rsi[-1]
        account_balance = self.equity
        risk_amount = account_balance * 0.01
        macro_tendency_bullish = self.data.Close[-1] > ema_value

        if macro_tendency_bullish and rsi_value < self.oversold:
            if not self.position:
                take_profit = self.data.Close[-1] * 1.03
                stop_loss = self.ema[-1] * 0.99
                price = self.data.Close[-1]
                risk_per_share = price - stop_loss
                position_size = min(risk_amount / risk_per_share, self.equity / price)
                position_size = max(1, round(position_size))
                self.buy(tp=take_profit, sl=stop_loss, size=position_size)

        elif not macro_tendency_bullish and rsi_value > self.overbought:
            if self.position:
                take_profit = self.data.Close[-1] * 0.97      
                stop_loss = self.ema[-1] * 1.01
                price = self.data.Close[-1]
                risk_per_share = stop_loss - price
                position_size = min(risk_amount / risk_per_share, self.equity / price)
                position_size = max(1, round(position_size))

df = df.loc['2019-01-01':'2024-01-01']

bt = Backtest(
    df, 
    EMARSI_Strategy, 
    cash=2000000, 
    commission=.002,
    trade_on_close=True,
    exclusive_orders=True
)

# Define the fitness function for PSO
def pso_fitness_function(params):
    ema_period, rsi_period, overbought, oversold = params
    results = bt.run(ema_period=int(ema_period), rsi_period=int(rsi_period), overbought=int(overbought), oversold=int(oversold))
    return -results['Return [%]'] + results['Max. Drawdown [%]']  # Minimize objective

# Define bounds for each parameter
lb = [50, 5, 70, 10]  # Lower bounds
ub = [200, 20, 90, 30]  # Upper bounds

# Run PSO
xopt, fopt = pso(pso_fitness_function, lb, ub)
print(f"Optimized parameters: EMA Period: {xopt[0]}, RSI Period: {xopt[1]}, Overbought: {xopt[2]}, Oversold: {xopt[3]}")

# Use xopt for final run
stats = bt.run(
    ema_period=int(xopt[0]), 
    rsi_period=int(xopt[1]), 
    overbought=int(xopt[2]), 
    oversold=int(xopt[3])
)

print(stats)
bt.plot(resample=None)
