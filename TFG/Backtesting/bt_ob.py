'''This bot will buy and sell in the OrderBlocks
First it will look in the EMA if we are bearish or bullish
If we are bullish it will take bullish OrderBlocks that are above the EMA
If we are bearish it will take bearish  OrderBlocks that are below the EMA
'''
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

# Function to calculate EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=period).mean()

# Function to detect pivot points (used to detect OBs)
def pivotid(df1, l, n1, n2): 
    # Check boundaries (avoid going out of the dataframe)
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0
    
    pividlow = True
    pividhigh = True
    # Checking local highs and lows within the given window (n1, n2)
    for i in range(l - n1, l + n2 + 1):
        if df1['Low'][l] > df1['Low'][i]:
            pividlow = False
        if df1['High'][l] < df1['High'][i]:
            pividhigh = False
    if pividlow:
        return 1  # Pivot Low (Bullish OB Candidate)
    elif pividhigh:
        return 2  # Pivot High (Bearish OB Candidate)
    else:
        return 0

# Function to identify Order Blocks (OBs)
def identify_ob(df):
    ob_zones = []
    for i in range(1, len(df) - 2):
        pivot = pivotid(df, i, 3, 3)

        if pivot == 1:  # Bullish OB
            ob_zones.append((i, 'bullish', df.Low[i], df.High[i], df.index[i]))  # (index, type, bottom of OB, top of OB, date)
        elif pivot == 2:  # Bearish OB
            ob_zones.append((i, 'bearish', df.High[i], df.Low[i], df.index[i]))  # (index, type, top of OB, bottom of OB, date)
    
    return ob_zones

# Function to determine if an Order Block is filled
def find_mitigation_time(ob_type, ob_bottom, ob_top, low_data, high_data, start_index):
    # Iterate through the data starting from the given index (next candle after the OB)
    for i in range(start_index + 1, len(low_data)):
        if ob_type == 'bullish' and low_data[i] <= ob_bottom:
            return i  # Return the index where it is filled
        elif ob_type == 'bearish' and high_data[i] >= ob_top:
            return i  # Return the index where it is filled
    return None

def is_pullback(df,ob_type,ob_index,mit_index): #Recorrer las velas entre el orderblock y la vela que lo mitigó
    pullback = True
    if ob_type == 'bullish':#Para determinar si la vela hizo un pullback, en algún momento el high de i tuvo que ser menor que el de i-1
        for i in range(ob_index,mit_index):
            if df['High'].iloc[i] > df['High'].iloc[i+1]:
                pullback = False

class OBStrategy(Strategy):
    ema_period = 200  # EMA period for trend detection
    risk_reward_ratio = 2  # Set a 1:3 risk/reward ratio
    sl_offset = 0.05  # Offset for stop-loss behind the bottom/top of the OB

    def init(self):
        # Register the 200-period EMA to assess trend direction
        self.ema = self.I(EMA, self.data.Close, self.ema_period)
        
        # Identify Order Blocks (OBs)
        self.obs = identify_ob(self.data.df)

        # Store order blocks that lead to trades and mitigation candles
        self.traded_obs = []
        self.filled_obs = set()  # Track indices of filled order blocks

    def next(self):
        # Get the most recent EMA value
        ema_value = self.ema[-1]
        
        # Determine the macro tendency: bullish if price is above EMA, bearish if below EMA
        macro_tendency_bullish = self.data.Close[-1] > ema_value
        
        # Get the current price
        current_price = self.data.Close[-1]

        # Loop through OBs and apply trading logic
        for ob in self.obs:
            ob_index, ob_type, ob_bottom, ob_top, ob_date = ob

            # Skip already filled order blocks
            if ob_index in self.filled_obs:
                continue

            # Only consider OBs that are still relevant (future ones)
            if ob_index >= len(self.data):
                continue

            # Find the mitigation time
            mitigation_time = find_mitigation_time(ob_type, ob_bottom, ob_top, self.data.Low, self.data.High, ob_index)
            
            # If no mitigation time found, continue
            if mitigation_time is None:
                continue
            
            # Mark this order block as filled
            self.filled_obs.add(ob_index)

            # Bullish scenario (Long Trade)
            if macro_tendency_bullish and ob_type == 'bullish' and ob_bottom > ema_value:
                # Buy if price is within the bullish OB (between top and bottom of the OB)
                if ob_bottom <= current_price <= ob_top and not self.position:
                    entry_price = current_price
                    stop_loss = ob_bottom * (1 - self.sl_offset)  # SL slightly below the bottom of the OB
                    take_profit = entry_price + ((entry_price - stop_loss) * self.risk_reward_ratio)

                    # Ensure valid order with SL < entry < TP and set entry explicitly as the limit
                    if stop_loss < entry_price < take_profit:
                        self.buy(sl=stop_loss, tp=take_profit, limit=entry_price, size=0.1)
                        # Store the order block creation date and the mitigation (entry) candle date
                        self.traded_obs.append((ob_type, ob_bottom, ob_top, ob_date, self.data.index[mitigation_time], self.data.Close[mitigation_time]))
                    
            # Bearish scenario (Short Trade)
            elif not macro_tendency_bullish and ob_type == 'bearish' and ob_bottom < ema_value:
                # Sell if price is within the bearish OB (between bottom and top of the OB)
                if ob_top <= current_price <= ob_bottom and not self.position:
                    entry_price = current_price
                    stop_loss = ob_top * (1 + self.sl_offset)  # SL slightly above the top of the OB
                    take_profit = entry_price - ((stop_loss - entry_price) * self.risk_reward_ratio)

                    # Ensure valid order with TP < entry < SL and set entry explicitly as the limit
                    if take_profit < entry_price < stop_loss:
                        self.sell(sl=stop_loss, tp=take_profit, limit=entry_price, size=0.1)
                        # Store the order block creation date and the mitigation (entry) candle date
                        self.traded_obs.append((ob_type, ob_bottom, ob_top, ob_date, self.data.index[mitigation_time], self.data.Close[mitigation_time]))

# Filter the data range to one year
df = df.loc['2022-01-01':'2023-01-01']

# Run the backtest
bt = Backtest(
    df, 
    OBStrategy, 
    cash=200_000, 
    commission=.002,
    trade_on_close=True,
    exclusive_orders=True
)
stats = bt.run()

# Print the order blocks that triggered trades and mitigating candle details
print("Order Blocks that Triggered Trades and their Mitigation Details:")
for ob in stats._strategy.traded_obs:
    ob_type, ob_bottom, ob_top, ob_date, mitigated_date, mitigated_price = ob
    print(f"OB Type: {ob_type}, OB Bottom: {ob_bottom}, OB Top: {ob_top}, OB Date: {ob_date}")
    print(f"Mitigated on: {mitigated_date}, Mitigation Price: {mitigated_price}\n")

# Plot the results
bt.plot(resample=None)


