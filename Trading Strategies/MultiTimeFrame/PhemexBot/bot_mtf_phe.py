import ccxt
import pandas as pd
import numpy as np
import key_file as k
import time
import schedule

# Initialize Phemex client with API keys
phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': k.key,
    'secret': k.secret
})

symbol = 'WIFUSDT'  # Adjust your trading pair
pos_size = 1.0  # Set your position size as a float
params = {'timeInForce': 'PostOnly'}
target_profit_pct = 3  # Target profit in percentage
max_loss_pct = -1  # Maximum allowed loss in percentage
ema_period = 100  # EMA period
sl_offset = 0.001  # Stop-loss offset
risk_reward_ratio = 3  # Risk/Reward ratio

# Function to fetch market data
def fetch_data(symbol, timeframe, limit=1000):
    ohlcv = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to calculate the EMA
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
        if df1['Low'].iloc[l] > df1['Low'].iloc[i]:
            pividlow = False
        if df1['High'].iloc[l] < df1['High'].iloc[i]:
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
def identify_ob(df, start_index=0):
    start_index = int(start_index)
    ob_signals = np.full(len(df), np.nan)
    top = np.full(len(df), np.nan)
    bottom = np.full(len(df), np.nan)
    mitigated_index = np.full(len(df), np.nan)
    ob_indices = np.full(len(df), np.nan)
    pullback_index = np.full(len(df), np.nan)
    
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
            if most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index,i, df):
                ob_signals[i] = 1
                top[i] = df['High'].iloc[i]
                bottom[i] = df['Low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i

        if pivot == -1:  # Bearish OB         
            if most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index,i, df):
                ob_signals[i] = -1
                top[i] = df['High'].iloc[i]
                bottom[i] = df['Low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i

    for i in np.where(~np.isnan(ob_signals))[0]:
        ob_type = ob_signals[i]
        inside_bar_check = True
        if ob_type == 1:  # Bullish OB
            for j in range(i + 1, len(df)):
                if df['Low'].iloc[j] <= top[i]:
                    if inside_bar_check:
                        inside_bar = is_inside_bar(i, j, df)
                        if not inside_bar:
                            inside_bar_check = False
                    if not inside_bar_check:
                        if is_pullback(i, j, df, ob_type):
                            mitigated_index[i] = j
                            break

        elif ob_type == -1:  # Bearish OB
            for j in range(i + 1, len(df)):
                if df['High'].iloc[j] >= bottom[i]:
                    if inside_bar_check:
                        inside_bar = is_inside_bar(i, j, df)
                        if not inside_bar:
                            inside_bar_check = False
                    if not inside_bar_check:
                        if is_pullback(i, j, df, ob_type):
                            mitigated_index[i] = j
                            break

    result_df = pd.DataFrame({
        'OB': ob_signals,
        'Top': top,
        'Bottom': bottom,
        'MitigatedIndex': mitigated_index,
        'OBIndex': ob_indices,
        'PBIndex': pullback_index
    })

    return result_df

# Function to check open positions and calculate PnL
def open_positions():
    params = {'type': 'swap', 'code': 'USDT'}
    balance = phemex.fetch_balance(params=params)
    positions = balance['info']['data']['positions']
    
    if len(positions) > 0:
        open_pos = positions[0]  # Assuming a single position
        size = float(open_pos['size'])  # Cast to float
        side = open_pos['side']  # Buy or Sell
        
        # Check if 'entryPrice' exists before accessing it
        if 'entryPrice' in open_pos:
            entry_price = float(open_pos['entryPrice'])  # Entry price of the position
        else:
            entry_price = None  # No entry price available
        
        return open_pos, size, side, entry_price
    return None, 0, None, None  # No position

# Function to calculate PnL
def calculate_pnl(entry_price, current_price, side):
    if side == 'Buy':
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        pnl_pct = ((entry_price - current_price) / entry_price) * 100
    return pnl_pct

# Function to check for open orders
def check_open_orders(symbol):
    open_orders = phemex.fetch_open_orders(symbol=symbol)
    return len(open_orders) > 0  # Returns True if there are open orders, False otherwise

# Main bot logic
def bot():
    df_HTF = fetch_data(symbol, '1h', 1000)  # Fetch 1h timeframe data
    df_LTF = fetch_data(symbol, '15m', 1000)  # Fetch 15m timeframe data
    
    # Calculate the EMA for trend detection on the HTF (6h)
    ema_HTF = EMA(df_HTF['Close'], ema_period)
    ema_value_HTF = ema_HTF.iloc[-1]
    
    # Identify order blocks on the 6h and 15m timeframes
    obs_HTF = identify_ob(df_HTF)
    obs_LTF = identify_ob(df_LTF)
    
    current_price = df_HTF['Close'].iloc[-1]
    macro_trend_bullish = current_price > ema_value_HTF
    
    # Check if we have an open position
    open_pos, size, side, entry_price = open_positions()
    in_position = size > 0

    # If in a position, check PnL and exit conditions
    if in_position:
        pnl_pct = calculate_pnl(entry_price, current_price, side)
        print(f"Current PnL: {pnl_pct}%")

        if pnl_pct >= target_profit_pct or pnl_pct <= max_loss_pct:
            print(f"Exiting position at {current_price} with PnL: {pnl_pct}%")
            if side == 'Buy':
                phemex.create_limit_sell_order(symbol, size, current_price, params)
            elif side == 'Sell':
                phemex.create_limit_buy_order(symbol, size, current_price, params)
        else:
            print(f"Holding position with PnL: {pnl_pct}%")
        return

    # Check if there are open orders for this symbol before placing new ones
    if check_open_orders(symbol):
        print(f"Open order exists for {symbol}. Skipping new order.")
        return

    # Now iterate through the HTF OBs and check for mitigation
    for ob_HTF in reversed(list(obs_HTF.itertuples())):
        if pd.isna(ob_HTF.OB):
            print(f"No valid order block found: {ob_HTF}")
            continue
        ob_type_HTF, ob_top_HTF, ob_bottom_HTF, ob_mit_index_HTF,ob_index_HTF = ob_HTF.OB, ob_HTF.Top, ob_HTF.Bottom, ob_HTF.MitigatedIndex,ob_HTF.OBIndex
        current_index = df_HTF.index[-1]
        
        # If the HTF OB is mitigated, check for LTF confirmation
        if current_index == ob_mit_index_HTF :  # If HTF OB is mitigated  
            if macro_trend_bullish:
                if ob_type_HTF == 1 and current_price < ob_top_HTF:  # Bullish OB
                    print(f"Bullish HTF OB mitigated at :{ob_HTF}, checking LTF OB")
                    for ob_LTF in reversed(list(obs_LTF.itertuples())):
                        ob_type_LTF, ob_top_LTF, ob_bottom_LTF = ob_LTF.OB, ob_LTF.Top, ob_LTF.Bottom

                        if ob_type_LTF == 1 and current_price > ob_top_LTF and ob_bottom_LTF >= ob_bottom_HTF and ob_top_LTF<= ob_top_HTF:  # Bullish LTF OB
                            print(f"Placing BUY order for bullish OB at {ob_top_LTF}, current price: {current_price}")
                            phemex.create_limit_buy_order(symbol, pos_size, ob_top_LTF, params)
                            return  # Exit after placing the order
            elif not macro_trend_bullish:
                if ob_type_HTF == -1 and current_price > ob_bottom_HTF:  # Bearish OB
                    print(f"Bearish HTF OB mitigated at :{ob_HTF}, checking LTF OB")
                    for ob_LTF in reversed(list(obs_LTF.itertuples())):
                        ob_type_LTF, ob_top_LTF, ob_bottom_LTF = ob_LTF.OB, ob_LTF.Top, ob_LTF.Bottom

                        if ob_type_LTF == -1 and current_price < ob_bottom_LTF and ob_bottom_LTF >= ob_bottom_HTF and ob_top_LTF<= ob_top_HTF:  # Bearish LTF OB
                            print(f"Placing SELL order for bearish OB at {ob_bottom_LTF}, current price: {current_price}")
                            phemex.create_limit_sell_order(symbol, pos_size, ob_bottom_LTF, params)
                            return  # Exit after placing the order
        else:
            print(f'Havent mitigated the HTF OB: {ob_HTF}')
            return

# Schedule the bot to run every 15 seconds
schedule.every(15).seconds.do(bot)

# Run the bot in a loop
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
