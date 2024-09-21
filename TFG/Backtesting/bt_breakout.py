import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\BTC-1h-100wks-data.csv')

# Ensure 'Date' column exists and is properly formatted
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Support and resistance calculation functions
def is_support(df, l, n1, n2):
    for i in range(l - n1 + 1, l + 1):
        if df.Low[i] > df.Low[i - 1]:
            return 0
    for i in range(l + 1, l + n2 + 1):
        if df.Low[i] < df.Low[i - 1]:
            return 0
    return 1

def is_resistance(df, l, n1, n2):
    for i in range(l - n1 + 1, l + 1):
        if df.High[i] < df.High[i - 1]:
            return 0
    for i in range(l + 1, l + n2 + 1):
        if df.High[i] > df.High[i - 1]:
            return 0
    return 1

# Strategy Class
class SupportResistanceWithMacroStrategy(Strategy):
    n1 = 3  # Number of bars to the left for support/resistance
    n2 = 3  # Number of bars to the right for support/resistance
    ema_period = 50  # EMA period for macro tendency
    buffer_pct = 0.1  # Buffer percentage for support/resistance (e.g., 10% above/below)
    risk_reward_ratio = 3
    sl_offset = 0.001
    
    def init(self):
        # Register the support and resistance indicators separately 
        self.supports = self.I(self.calculate_support)
        self.resistances = self.I(self.calculate_resistance)
        
        # Register the EMA to assess macro trend
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, min_periods=self.ema_period).mean(), self.data.Close)
        
        # Track the last support/resistance used for a trade and whether a trade has been placed
        self.used_supports = set()
        self.used_resistances = set()

    def calculate_support(self):
        supports = np.full(len(self.data), np.nan)
        for row in range(self.n1, len(self.data) - self.n2):
            if is_support(self.data, row, self.n1, self.n2):
                supports[row] = self.data.Low[row]
        return supports

    def calculate_resistance(self):
        resistances = np.full(len(self.data), np.nan)
        for row in range(self.n1, len(self.data) - self.n2):
            if is_resistance(self.data, row, self.n1, self.n2):
                resistances[row] = self.data.High[row]
        return resistances

    def next(self):
        # Get the most recent support and resistance values
        current_index = len(self.data) - 1
        support = self.supports[current_index]
        resistance = self.resistances[current_index]
        
        # Get the most recent EMA value (macro trend indicator)
        ema_value = self.ema[-1]
        
        # Determine the macro tendency: bullish if price is above EMA, bearish if below EMA
        macro_tendency_bullish = self.data.Close[-1] > ema_value
        
        # Define the buffer range for buying (support to support + buffer)
        support_buffer = support * (1 + self.buffer_pct) if support else None

        # Define the buffer range for selling (resistance to resistance - buffer)
        resistance_buffer = resistance * (1 - self.buffer_pct) if resistance else None

        ### Buying logic (Bullish market)
        if macro_tendency_bullish and support and support not in self.used_supports:
            # Buy when price is within the support and buffer range
            if support <= self.data.Low[-1] <= support_buffer:
                entry_price = support
                stop_loss = support * (1 - self.sl_offset)
                take_profit = entry_price + ((entry_price - stop_loss) * self.risk_reward_ratio)
                self.buy(sl=stop_loss, tp=take_profit, size=0.1, limit=entry_price)
                self.used_supports.add(support)  # Mark this support as used

        ### Selling logic (Bearish market)
        elif not macro_tendency_bullish and resistance and resistance not in self.used_resistances:
            # Sell when price is within the resistance and buffer range
            if resistance_buffer <= self.data.High[-1] <= resistance:
                entry_price = resistance
                stop_loss = resistance * (1 + self.sl_offset)
                take_profit = entry_price - ((stop_loss - entry_price) * self.risk_reward_ratio)
                self.sell(sl=stop_loss, tp=take_profit, size=0.1, limit=entry_price)
                self.used_resistances.add(resistance)  # Mark this resistance as used

# Filter the data to a specific date range
df = df.loc['2022-01-01':'2023-01-01']

# Perform the backtest using BTCUSD data
bt = Backtest(
    df, 
    SupportResistanceWithMacroStrategy, 
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
