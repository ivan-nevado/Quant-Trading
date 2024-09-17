'''This bot will look at the ema of 10 periods and see if we are bullish or bearish
If we are bullish and on the rsi we are oversold(above the 70 value) we buy
If we are bearish and on the rsi we are overbought(above the 30 value) we sell'''

import pandas as pd
from backtesting import Backtest, Strategy

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\BTC-1h-100wks-data.csv')

# Ensure 'Date' column exists and is properly formatted
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Function to calculate RSI (handling the backtesting.py internal data type)
def RSI(arr, period=30):
    series = pd.Series(arr)  # Convert backtesting.py internal _Array to pandas.Series
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Strategy Class
class EMARSI_Strategy(Strategy):
    ema_period = 30  # EMA period for trend detection
    rsi_period = 30  # RSI period for overbought/oversold
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
        
        # Determine the macro tendency: bullish if price is above EMA, bearish if below EMA
        macro_tendency_bullish = self.data.Close[-1] > ema_value

        # Trading logic based on EMA and RSI:
        # Buy if bullish trend and RSI indicates overbought (RSI > 70)
        if macro_tendency_bullish and rsi_value > self.overbought:
            if not self.position:
                self.buy(size=0.1)

        # Sell if bearish trend and RSI indicates oversold (RSI < 30)
        elif not macro_tendency_bullish and rsi_value < self.oversold:
            if self.position:
                self.sell(size=0.1)

# Filter the dataset for a specific date range
df = df.loc['2022-01-01':'2023-01-01']

# Perform the backtest using BTCUSD data
bt = Backtest(
    df, 
    EMARSI_Strategy, 
    cash=200_000, 
    commission=.002,
    trade_on_close=True,
    exclusive_orders=True
)

# Run the backtest without optimization first
stats = bt.run()

# Print the stats
print(stats)

# Plot the results without resampling
bt.plot(resample=None)
