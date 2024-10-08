'''The idea of the bot is this:
1. It will check for the tendency on the HTF (6h timeframe)
2. If it is bullish it will look for bullish OB on the HTF (6h timeframe)
3. It has to wait for price to mitigate the 6h OB
4. When the 6h OB is mitigated it will check for the MTF tendency (15m timeframe)
5. If the 15m timeframe is bullish it will look for a bullish orderblock mitigation
'''
import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load 6h data
df_HTF = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Iván Quant Trading\TFG\BTC-1h-300wks-data.csv')
df_HTF['Date'] = pd.to_datetime(df_HTF['Date'])
df_HTF.set_index('Date', inplace=True)

# Load 6h data
df_LTF = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Iván Quant Trading\TFG\BTC-15m-300wks-data.csv')
df_LTF['Date'] = pd.to_datetime(df_LTF['Date'])
df_LTF.set_index('Date', inplace=True)


# Function to detect pivot points (used to detect OBs)
def pivotid(df1, l, n1, n2): 
    # Check boundaries (avoid going out of the dataframe)
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0
    
    pividlow = True
    pividhigh = True
    # Checking local highs and lows within the given window (n1, n2)
    for i in range(l - n1, l + n2 + 1):
        if df1['Low'].iloc[l] > df1['Low'].iloc[i]:
            pividlow = False
        if df1['High'].iloc[l] < df1['High'].iloc[i]:
            pividhigh = False
            
    if pividlow :
        if (df1["High"].iloc[l] < df1["Low"].iloc[l+2]) and (df1["Close"].iloc[l+1] > df1["Open"].iloc[l+1]):
            return 1  # Pivot Low (Bullish OB Candidate)
    elif pividhigh :
        if (df1["Low"].iloc[l] > df1["High"].iloc[l+2]) and (df1["Close"].iloc[l+1] < df1["Open"].iloc[l+1]):
            # print(f'Pivot High detected at index {l}, value: {df1["High"].iloc[l]}')
            return -1  # Pivot High (Bearish OB Candidate)
    else:
        return 0 

# Function to identify Order Blocks (OBs)
def identify_ob(df,start_index = 0):
    start_index = int(start_index)
    ob_signals = np.full(len(df), np.nan)
    top = np.full(len(df), np.nan)
    bottom = np.full(len(df), np.nan)
    mitigated_index = np.full(len(df), np.nan)
    ob_indices = np.full(len(df), np.nan)
    pullback_index = np.full(len(df), np.nan)
    # print(f'start_index: {start_index}, df: {df}')
    
    

    def is_inside_bar(prev_ob_index,new_ob_index, df):
        return df['High'].iloc[prev_ob_index] >= df['High'].iloc[new_ob_index] and df['Low'].iloc[prev_ob_index] <= df['Low'].iloc[new_ob_index]

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
                    if df['High'].iloc[j] < df['High'].iloc[j - 1]:
                        return True
            elif ob_type == -1:  # Bearish OB
                if inside_bar_check:
                    inside_bar = is_inside_bar(ob_index, j, df)
                    if not inside_bar:
                        inside_bar_check = False
                if not inside_bar_check:
                    # Pullback happens when the low of the current candle is higher than the previous candle
                    if df['Low'].iloc[j] > df['Low'].iloc[j - 1]:
                        return True
        return False
    
    most_recent_ob_index = None  # Variable to keep track of the most recent OB

    
    for i in range(start_index, len(df) - 2):
        pivot = pivotid(df, i, 3, 3)

        if pivot == 1:  # Bullish OB         
            if  most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index,i, df):
                ob_signals[i] = 1
                top[i] = df['High'].iloc[i]
                bottom[i] = df['Low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i
                # print(f'Pivot Low detected at index {i}, value: {df["Low"].iloc[i]}')

                        
        if pivot == -1:  # Bullish OB         
            if  most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index,i, df):
                ob_signals[i] = -1
                top[i] = df['High'].iloc[i]
                bottom[i] = df['Low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i
                # print(f'Pivot High detected at index {i}, value: {df["High"].iloc[i]}')

       
#Tiene primero que esperar a que los candles posteriores salgan del orderblock en el caso de que sean inside bars
#Una vez salen tienen que esperar un pullback y ya no mirar lo del inside bar
    for i in np.where(~np.isnan(ob_signals))[0]:
        ob_type = ob_signals[i]
        inside_bar_check = True
        if ob_type == 1:  # Bullish OB
            # Look for mitigation starting from the next candle (i + 1)
            for j in range(i + 1, len(df)):
                # Check if the current candle mitigates the OB
                if df['Low'].iloc[j] <= top[i]:
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
                if df['High'].iloc[j] >= bottom[i]:
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
        pullback_index[i] = np.nan  # Initialize pullback index as NaN
        if pd.notna(mitigated_index[i]):  # Check if mitigated_index[i] is not NaN
            ob_index = int(ob_indices[i])
            mit_index = int(mitigated_index[i])
            
            if ob_type == 1:  # Bullish OB
                highest_high = df['High'].iloc[ob_index]
                for j in range(ob_index + 1, mit_index):
                    if df['High'].iloc[j] > highest_high:
                        highest_high = df['High'].iloc[j]
                        pullback_index[i] = j
            elif ob_type == -1:  # Bearish OB
                lowest_low = df['Low'].iloc[ob_index]
                for j in range(ob_index + 1, mit_index):
                    if df['Low'].iloc[j] < lowest_low:
                        lowest_low = df['Low'].iloc[j]
                        pullback_index[i] = j
                        
    result_df = pd.DataFrame({
        'OB': ob_signals,
        'Top': top,
        'Bottom': bottom,
        'MitigatedIndex': mitigated_index,
        'OBIndex': ob_indices,
        'PBIndex': pullback_index
    })

    return result_df

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


class MultiTimeframeOBStrategy(Strategy):
    data_15m = None  # Declare 15m data as a class variable
    min_ob_height = 100
    ema_period = 100
    risk_reward_ratio = 3
    sl_offset = 0.001

    def init(self):
        self.data_HTF = self.data.df
        self.data_LTF = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Iván Quant Trading\TFG\BTC-15m-300wks-data.csv',index_col='Date',parse_dates=True)
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, adjust=False).mean(), self.data_HTF.Close)
        
        start_date = max(self.data_HTF.index[0], self.data_LTF.index[0])
        end_date = min(self.data_HTF.index[-1], self.data_LTF.index[-1])
        self.data_HTF = self.data_HTF.loc[start_date:end_date]
        self.data_LTF = self.data_LTF.loc[start_date:end_date]


        
        self.obs_HTF = identify_ob(self.data_HTF)
        self.obs_LTF = identify_ob(self.data_LTF)
        
        # LTF OB detection (1H timeframe)
        self.current_HTF_ob = None
        self.invalidate_HTF_ob = set()
        # Create a mapping between HTF and LTF indices
        self.htf_to_ltf_index_map = {i: i*4 for i in range(len(self.data))}

    def next(self):
        
        current_HTF_index = len(self.data) - 1
        last_HTF_ob = None
        end_current_ltf_index = None
        macro_trend_bullish = self.data.Close[-1] > self.ema[-1]
        
        valid_HTF_obs = self.obs_HTF[(self.obs_HTF.OBIndex <= current_HTF_index) & (~self.obs_HTF.index.isin(self.invalidate_HTF_ob))]
        for _, ob_HTF in valid_HTF_obs.iterrows():
            ob_type_HTF = ob_HTF.OB
            ob_top_HTF = ob_HTF.Top
            ob_bottom_HTF = ob_HTF.Bottom
            ob_mit_index_HTF = ob_HTF.MitigatedIndex
            ob_index_HTF = ob_HTF.OBIndex
            ob_pb_index_HTF = ob_HTF.PBIndex
            
            self.current_HTF_ob = ob_HTF
            if current_HTF_index == ob_mit_index_HTF:
                if macro_trend_bullish:
                    if ob_type_HTF == 1:
                        start_LTF_index = self.htf_to_ltf_index_map[ob_index_HTF]
                        current_LTF_index = self.htf_to_ltf_index_map[current_HTF_index]
                        end_current_ltf_index = current_LTF_index + 3
                        valid_LTF_obs = self.obs_LTF[(self.obs_LTF.OBIndex >= start_LTF_index) & 
                                                     (self.obs_LTF.OBIndex <= end_current_ltf_index)]
                        valid_LTF_range = True
                        

                        if valid_LTF_obs.empty:
                            valid_LTF_range = False
                        
                        while valid_LTF_range and current_LTF_index <= end_current_ltf_index:
                        
                            for _, ob_LTF in valid_LTF_obs.iterrows():
                                ob_type_LTF = ob_LTF.OB
                                ob_top_LTF = ob_LTF.Top
                                ob_bottom_LTF = ob_LTF.Bottom
                                ob_mit_index_LTF = ob_LTF.MitigatedIndex
                                ob_index_LTF = ob_LTF.OBIndex
                                ob_pb_index_LTF = ob_LTF.PBIndex
                                
                                if current_LTF_index < len(self.data_LTF):
                                    if self.data_LTF.Low.iloc[current_LTF_index] < ob_bottom_HTF:
                                        self.invalidate_HTF_ob.add(ob_index_HTF)
                                        valid_LTF_range = False
                                        break
                                else:
                                    valid_LTF_range = False
                                    break
                                
                                if current_LTF_index == ob_mit_index_LTF:
                                    if ob_type_LTF == 1:
                                        entry_price = ob_top_LTF
                                        stop_loss = ob_bottom_LTF * (1 - self.sl_offset)
                                        take_profit = entry_price + ((entry_price - stop_loss) * self.risk_reward_ratio)

                                        if stop_loss < entry_price < take_profit:
                                            self.buy(sl=stop_loss, tp=take_profit, size=0.1, limit=entry_price)
                                            print(f"ob index: {ob_index_LTF}, ob top:{ob_top_LTF}, ob bottom: {ob_bottom_LTF},Entry price: {entry_price}, Stop loss: {stop_loss},take profit: {take_profit}")
                                            
                            current_LTF_index += 1
                
                elif not macro_trend_bullish:                    
                    if ob_type_HTF == -1:
                        start_LTF_index = self.htf_to_ltf_index_map[ob_index_HTF]
                        current_LTF_index = self.htf_to_ltf_index_map[current_HTF_index]
                        end_current_ltf_index = current_LTF_index + 3
                        valid_LTF_obs = self.obs_LTF[(self.obs_LTF.OBIndex >= start_LTF_index) & 
                                                     (self.obs_LTF.OBIndex <= end_current_ltf_index)]
                        valid_LTF_range = True

                        if valid_LTF_obs.empty:
                            valid_LTF_range = False
                    
                        while valid_LTF_range and current_LTF_index <= end_current_ltf_index:

                            for _, ob_LTF in valid_LTF_obs.iterrows():
                                ob_type_LTF = ob_LTF.OB
                                ob_top_LTF = ob_LTF.Top
                                ob_bottom_LTF = ob_LTF.Bottom
                                ob_mit_index_LTF = ob_LTF.MitigatedIndex
                                ob_index_LTF = ob_LTF.OBIndex
                                ob_pb_index_LTF = ob_LTF.PBIndex
                            
                                if current_LTF_index < len(self.data_LTF):
                                    if self.data_LTF.High.iloc[current_LTF_index] > ob_top_HTF:
                                        self.invalidate_HTF_ob.add(ob_index_HTF)
                                        valid_LTF_range = False
                                        break
                                else:
                                    valid_LTF_range = False
                                    break  # Exit the loop if index is out of bounds
                                
                                #The pullback index loop is bad coded
                        
                                if current_LTF_index == ob_mit_index_LTF:
                                    if ob_type_LTF == -1:
                                        entry_price = ob_bottom_LTF
                                        stop_loss = ob_top_LTF * (1 + self.sl_offset)
                                        take_profit = entry_price - ((stop_loss - entry_price) * self.risk_reward_ratio)


                                        if stop_loss > entry_price > take_profit:
                                            self.sell(sl=stop_loss, tp=take_profit, size=0.1, limit=entry_price)
                                            print(f"ob index: {ob_index_LTF}, ob top:{ob_top_LTF}, ob bottom: {ob_bottom_LTF},Entry price: {entry_price}, Stop loss: {stop_loss},take profit: {take_profit}")
                                           

                            current_LTF_index += 1
                        
                        
# Filter the data range to one year
df_HTF = df_HTF.loc['2023-01-01':'2024-01-01']

# Run the backtest
bt = Backtest(df_HTF, MultiTimeframeOBStrategy, cash=1000000, trade_on_close=False, exclusive_orders=False)
stats = bt.run()
print(stats)
bt.plot()
