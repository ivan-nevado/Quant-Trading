'''This bot will look at the ema of 150 periods and see if we are bullish or bearish
If we are bullish and on the rsi we are oversold(below the 30 value) we buy
If we are bearish and on the rsi we are overbought(above the 70 value) we sell'''

import pandas as pd
from backtesting import Backtest, Strategy

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Iván Quant Trading\TFG\BTC-6h-300wks-data.csv')

# Ensure 'Date' column exists and is properly formatted
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Function to calculate RSI (handling the backtesting.py internal data type)
def RSI(arr, period):
    series = pd.Series(arr)  # Convert backtesting.py internal _Array to pandas.Series
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Strategy Class
class EMARSI_Strategy(Strategy):
    ema_period = 150  # EMA period for trend detection
    rsi_period = 10  # RSI period for overbought/oversold
    overbought = 70  # Overbought threshold for RSI
    oversold = 30  # Oversold threshold for RSI

    def init(self):
        # Register the 10-period EMA to assess trend direction
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, min_periods=self.ema_period).mean(), self.data.Close)
        
        # Register the RSI indicator
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

    def next(self):
        # Get the most recent EMA and RSI values
        ema_value = self.ema[-1]
        rsi_value = self.rsi[-1]
        account_balance = self.equity
        risk_amount = account_balance * 0.01
        # Determine the macro tendency: bullish if price is above EMA, bearish if below EMA
        macro_tendency_bullish = self.data.Close[-1] > ema_value

        # Trading logic based on EMA and RSI:
        # Buy if bullish trend and RSI indicates oversold (RSI < 30)
        if macro_tendency_bullish and rsi_value < self.oversold:
            if not self.position:
                take_profit = self.data.Close[-1] * 1.03
                stop_loss = self.ema[-1] * 0.99
                price = self.data.Close[-1]
                risk_per_share = price - stop_loss
                position_size = min(risk_amount / risk_per_share, self.equity / price)
                position_size = max(1, round(position_size))  # Ensure at least 1 unit
                self.buy(tp=take_profit, sl=stop_loss, size=position_size)
                

        # Sell if bearish trend and RSI indicates overbought (RSI > 70)
        elif not macro_tendency_bullish and rsi_value > self.overbought:
            if self.position:
                take_profit = self.data.Close[-1] * 0.97      
                stop_loss = self.ema[-1] * 1.01
                price = self.data.Close[-1]
                risk_per_share = stop_loss - price
                position_size = min(risk_amount / risk_per_share, self.equity / price)
                position_size = max(1, round(position_size))  # Ensure at least 1 unit
                self.sell(tp=take_profit, sl=stop_loss, size=position_size)
# Filter the dataset for a specific date range
df = df.loc['2019-01-01':'2024-01-01']

# Perform the backtest using BTCUSD data
bt = Backtest(
    df, 
    EMARSI_Strategy, 
    cash=2000000, 
    commission=.002,
    trade_on_close=True,
    exclusive_orders=True
)

results = bt.optimize(
     rsi_period=[10, 15, 20, 25, 30],
     ema_period=[50, 75, 100, 125, 150, 175, 200],
     overbought = [60,65,70,75,80],
     oversold = [20,25,30,35,40],
     maximize='Return [%]'
)


# Print the optimized parameters
print(f"Optimized parameters:")
print(f"Best RSi period: {results._strategy.rsi_period}")
print(f"Best EMA period: {results._strategy.ema_period}")
print(f'overbough value: {results._strategy.overbought}')
print(f'oversold value: {results._strategy.oversold}')


stats = bt.run()
print(stats)
# Plot the results without resampling
bt.plot(resample=None)