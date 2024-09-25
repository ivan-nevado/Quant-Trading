'''This bot will buy and sell in the OrderBlocks
First it will look in the EMA if we are bearish or bullish
If we are bullish it will take bullish OrderBlocks that are above the EMA
If we are bearish it will take bearish  OrderBlocks that are below the EMA
'''
import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\BTC-1h-100wks-data.csv')

# Ensure 'Date' column exists and is properly formatted
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

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
            
    if pividlow :
        if (df1["High"].iloc[l] < df1["Low"].iloc[l+2]) and (df1["Close"].iloc[l+1] > df1["Open"].iloc[l+1]):
            return 1  # Pivot Low (Bullish OB Candidate)
    elif pividhigh :
        if (df1["Low"].iloc[l] > df1["High"].iloc[l+2]) and (df1["Close"].iloc[l+1] < df1["Open"].iloc[l+1]):
            return -1  # Pivot High (Bearish OB Candidate)
    else:
        return 0

# Function to identify Order Blocks (OBs)
def identify_ob(df):
    ob_signals = np.full(len(df), np.nan)
    top = np.full(len(df), np.nan)
    bottom = np.full(len(df), np.nan)
    mitigated_index = np.full(len(df), np.nan)
    ob_indices = np.full(len(df), np.nan)
    pullback_index = np.full(len(df), np.nan)
    

    def is_inside_bar(prev_ob_index,new_ob_index, df):
        return df['High'][prev_ob_index] >= df['High'][new_ob_index] and df['Low'][prev_ob_index] <= df['Low'][new_ob_index]

    def is_pullback(ob_index, mit_index, df, ob_type):
        inside_bar_check = True
        start_index = ob_index + 2  # Start after the gap candle
        for j in range(start_index, mit_index + 1):
            if ob_type == 1:  # Bullish OB
                if inside_bar_check:
                    inside_bar = is_inside_bar(ob_index, j, df)
                    if not inside_bar:
                        inside_bar_check = False
                if not inside_bar_check:
                    # Pullback happens when the high of the current candle is lower than the previous candle
                    if df['High'][j] < df['High'][j - 1]:
                        return True
            elif ob_type == -1:  # Bearish OB
                if inside_bar_check:
                    inside_bar = is_inside_bar(ob_index, j, df)
                    if not inside_bar:
                        inside_bar_check = False
                if not inside_bar_check:
                    # Pullback happens when the low of the current candle is higher than the previous candle
                    if df['Low'][j] > df['Low'][j - 1]:
                        return True
        return False
    
    most_recent_ob_index = None  # Variable to keep track of the most recent OB

    
    for i in range(1, len(df) - 2):
        pivot = pivotid(df, i, 3, 3)
    
        if pivot == 1:  # Bullish OB         
            if  most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index,i, df):
                ob_signals[i] = 1
                top[i] = df['High'][i]
                bottom[i] = df['Low'][i]
                ob_indices[i] = i
                most_recent_ob_index = i
                        
        if pivot == -1:  # Bullish OB         
            if  most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index,i, df):
                ob_signals[i] = -1
                top[i] = df['High'][i]
                bottom[i] = df['Low'][i]
                ob_indices[i] = i
                most_recent_ob_index = i
       
#Tiene primero que esperar a que los candles posteriores salgan del orderblock en el caso de que sean inside bars
#Una vez salen tienen que esperar un pullback y ya no mirar lo del inside bar
    for i in np.where(~np.isnan(ob_signals))[0]:
        ob_type = ob_signals[i]
        inside_bar_check = True
        if ob_type == 1:  # Bullish OB
            # Look for mitigation starting from the next candle (i + 1)
            for j in range(i + 1, len(df)):
                # Check if the current candle mitigates the OB
                if df['Low'][j] <= top[i]:
                    # Ensure mitigation was done with a pullback and not an inside bar
                    if inside_bar_check:
                        inside_bar = is_inside_bar(i, j, df)
                        if not inside_bar:
                            inside_bar_check = False
                    if not inside_bar_check:
                        if is_pullback(i, j, df, ob_type):
                            mitigated_index[i] = j
                            break  # Once mitigated, no need to check further candles

        elif ob_type == -1:  # Bearish OB
            # Look for mitigation starting from the next candle (i + 1)
            for j in range(i + 1, len(df)):
                # Check if the current candle mitigates the OB
                if df['High'][j] >= bottom[i]:
                    # Ensure mitigation was done with a pullback and not an inside bar
                    if inside_bar_check:
                        inside_bar = is_inside_bar(i, j, df)
                        if not inside_bar:
                            inside_bar_check = False
                    if not inside_bar_check:
                        if is_pullback(i, j, df, ob_type) and not is_inside_bar(i, j, df):
                            mitigated_index[i] = j
                            break  # Once mitigated, no need to check further candles


    # Save the index of the candle where the pullback starts
    for i in np.where(~np.isnan(ob_signals))[0]:
        ob_type = ob_signals[i]
        pullback_index = None  # Initial assumption for pullback start
        if pd.notna(mitigated_index[i]):  # Check if mitigated_index[i] is not NaN
            
            if ob_type == 1:  # Bullish OB
                highest_high = -np.inf  # Initialize with a very low value
                for j in range(i + 1, int(mitigated_index[i])):
                    if df['High'][j] > highest_high:
                        highest_high = df['High'][j]
                        pullback_index = j  # Update pullback index to the highest high

            elif ob_type == -1:  # Bearish OB
                lowest_low = np.inf  # Initialize with a very high value
                for j in range(i + 1, int(mitigated_index[i])):
                    if df['Low'][j] < lowest_low:
                        lowest_low = df['Low'][j]
                        pullback_index = j  # Update pullback index to the lowest low
        
    return pd.DataFrame({
        'OB': ob_signals,
        'Top': top,
        'Bottom': bottom,
        'MitigatedIndex': mitigated_index,
        'OBIndex': ob_indices,
        'PBIndex': pullback_index
    })
# Function to determine if an OB is filled
def is_ob_filled(ob_row, df, start_index, current_index):
    top = ob_row.Top
    bottom = ob_row.Bottom
    for i in range(start_index, current_index + 1):  # Only check up to the current index
        if (ob_row.OB == 1 and df['Low'].iloc[i] < bottom) or \
           (ob_row.OB == -1 and df['High'].iloc[i] > top):
            return True  # Return True if filled
    return False  # Return False if not filled

def detect_ema(ohlc: pd.DataFrame):
    ema = ohlc['Close'].ewm(span=100, adjust=False).mean()
    return ema

class OBStrategy(Strategy):
    min_ob_height = 100
    ema_period = 100
    risk_reward_ratio = 3
    sl_offset = 0.001
    def init(self): 
        self.obs = identify_ob(self.data.df)
        self.processed_obs = set()
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, adjust=False).mean(), self.data.Close)
        
    def next(self):
        current_index = len(self.data) - 1  # Get the current candle index
        # Filter OBs up to the current index
        valid_obs = self.obs[self.obs.OBIndex <= current_index]
        account_balance = self.equity
        risk_amount = account_balance * 0.01
        macro_trend_bullish = self.data.Low[-1] > self.ema[-1]
        for ob in valid_obs.itertuples():
            ob_type, ob_top, ob_bottom, ob_mit_index, ob_index = ob.OB, ob.Top, ob.Bottom, ob.MitigatedIndex, ob.OBIndex
            
             # Check if the OB height meets the minimum requirement
            ob_height = abs(ob_top - ob_bottom)
            if ob_height < self.min_ob_height:
                continue

            # Ensure indices are valid before proceeding
            if is_ob_filled(ob, self.data.df, int(ob_index) + 1, current_index):
                continue
            if pd.isna(ob_mit_index):
                continue  
            
            if ob_mit_index<current_index:
                continue
            ob_mit_index = int(ob_mit_index)  # Convert to integer index
            if macro_trend_bullish:
                if ob_type == 1:
                    # Check if the current candle is the mitigation index
                    if current_index == ob_mit_index:
                        # Check if the current price has touched or crossed the OB top
                        entry_price = ob_top  # Use the OB top as the entry price
                        stop_loss = ob_bottom * (1 - self.sl_offset)
                        take_profit = entry_price + ((entry_price - stop_loss) * self.risk_reward_ratio)
                        risk_per_share = entry_price - stop_loss
                        position_size = min(risk_amount / risk_per_share, self.equity / entry_price)
                        position_size = max(1, round(position_size))  # Ensure at least 1 unit

                    
                        if stop_loss < entry_price < take_profit:
                            self.buy(sl=stop_loss, tp=take_profit, size=0.1, limit=entry_price)
                            print(f'current index: {current_index}, entry price: {entry_price}, ob top: {ob_top}, ob_bottom: {ob_bottom},ob_index: {ob_index}')
                            self.processed_obs.add(int(ob_index))  # Mark this OB as processed
                        
            elif not macro_trend_bullish:
                if ob_type == -1:
                    if current_index == ob_mit_index:

                        entry_price = ob_bottom
                        stop_loss = ob_top * (1 + self.sl_offset)
                        take_profit = entry_price - ((stop_loss - entry_price) * self.risk_reward_ratio)
                        risk_per_share = stop_loss - entry_price
                        position_size = min(risk_amount / risk_per_share, self.equity / entry_price)
                        position_size = max(1, round(position_size))  # Ensure at least 1 unit

                        if stop_loss > entry_price > take_profit:
                            self.sell(sl=stop_loss, tp=take_profit, size=0.1, limit=entry_price)
                            print(f'current index: {current_index}, entry price: {entry_price}, ob top: {ob_top}, ob_bottom: {ob_bottom}, ob_index: {ob_index}')
                            self.processed_obs.add(int(ob_index))  # Mark this OB as processed

# Filter the data range to one year
df = df.loc['2023-06-01':'2023-12-01']

# Run the backtest

bt = Backtest(df, OBStrategy, cash=20000000, commission=.002, trade_on_close=False, exclusive_orders=False)  

# Run and plot the backtest
stats = bt.run()
print(stats)
bt.plot()
  