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
target_profit_pct = 8  # Target profit in percentage
max_loss_pct = -9  # Maximum allowed loss in percentage
ema_period = 100  # EMA period

# Function to fetch market data
def fetch_data(symbol, timeframe, limit=1000):
    ohlcv = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to fetch ask and bid prices
def ask_bid():
    ob = phemex.fetch_order_book(symbol)
    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    return ask, bid

# Function to calculate the EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=period).mean()

# Function to identify Order Blocks (OBs)
def identify_ob(df):
    ob_signals = np.full(len(df), np.nan)
    top = np.full(len(df), np.nan)
    bottom = np.full(len(df), np.nan)
    mitigated_index = np.full(len(df), np.nan)
    ob_indices = np.full(len(df), np.nan)

    def pivotid(df1, l, n1, n2):
        if l - n1 < 0 or l + n2 >= len(df1):
            return 0
        pividlow = True
        pividhigh = True
        for i in range(l - n1, l + n2 + 1):
            if df1['Low'][l] > df1['Low'][i]:
                pividlow = False
            if df1['High'][l] < df1['High'][i]:
                pividhigh = False
        if pividlow:
            return 1  # Bullish OB Candidate
        elif pividhigh:
            return -1  # Bearish OB Candidate
        else:
            return 0

    def is_inside_bar(prev_ob_index, new_ob_index, df):
        return df['High'][prev_ob_index] >= df['High'][new_ob_index] and df['Low'][prev_ob_index] <= df['Low'][new_ob_index]

    def is_pullback(ob_index, mit_index, df, ob_type):
        for j in range(ob_index + 1, mit_index):
            if ob_type == 1 and df['High'][j] < df['High'][j-1]:
                return True
            elif ob_type == -1 and df['Low'][j] > df['Low'][j-1]:
                return True
        return False

    most_recent_ob_index = None
    for i in range(1, len(df) - 2):
        pivot = pivotid(df, i, 3, 3)

        if pivot == 1:  # Bullish OB
            if most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index, i, df):
                ob_signals[i] = 1
                top[i] = df['High'].iloc[i]
                bottom[i] = df['Low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i

        if pivot == -1:  # Bearish OB
            if most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index, i, df):
                ob_signals[i] = -1
                top[i] = df['High'].iloc[i]
                bottom[i] = df['Low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i

    for i in np.where(~np.isnan(ob_signals))[0]:
        ob_type = ob_signals[i]
        if ob_type == 1:  # Bullish OB
            for j in range(i + 1, len(df)):
                if df['Low'].iloc[j] <= top[i]:
                    if is_pullback(i, j, df, ob_type):
                        mitigated_index[i] = j
                        break
        elif ob_type == -1:  # Bearish OB
            for j in range(i + 1, len(df)):
                if df['High'].iloc[j] >= bottom[i]:
                    if is_pullback(i, j, df, ob_type):
                        mitigated_index[i] = j
                        break

    return pd.DataFrame({
        'OB': ob_signals,
        'Top': top,
        'Bottom': bottom,
        'MitigatedIndex': mitigated_index,
        'OBIndex': ob_indices
    })

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
    df = fetch_data(symbol, '6h', 1000)
    ask, bid = ask_bid()
    obs = identify_ob(df)
    current_price = df['Close'].iloc[-1]
    ema = EMA(df['Close'], ema_period)
    ema_value = ema.iloc[-1]

    macro_trend_bullish = current_price > ema_value

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
                phemex.create_limit_sell_order(symbol, size, ask, params)
            elif side == 'Sell':
                phemex.create_limit_buy_order(symbol, size, bid, params)
        else:
            print(f"Holding position with PnL: {pnl_pct}%")
        return

    # Check if there are open orders for this symbol before placing new ones
    if check_open_orders(symbol):
        print(f"Open order exists for {symbol}. Skipping new order.")
        return

    # Loop through identified OBs in reverse order (most recent first)
    for ob in reversed(list(obs.itertuples())):
        ob_type, ob_top, ob_bottom, ob_mit_index = ob.OB, ob.Top, ob.Bottom, ob.MitigatedIndex

        # Place a BUY order for the most recent bullish OB
        if ob_type == 1 and current_price > ob_top and macro_trend_bullish and current_price > ema_value:
            print(f'Placing BUY order for bullish OB at {ob_top}, current price: {current_price}, EMA: {ema_value}')
            phemex.create_limit_buy_order(symbol, pos_size, ob_top, params)
            break  # Exit loop after placing the order

        # Place a SELL order for the most recent bearish OB
        elif ob_type == -1 and current_price < ob_bottom and not macro_trend_bullish and current_price < ema_value:
            print(f'Placing SELL order for bearish OB at {ob_bottom}, current price: {current_price}')
            phemex.create_limit_sell_order(symbol, pos_size, ob_bottom, params)
            break  # Exit loop after placing the order

# Schedule the bot to run every 15 seconds
schedule.every(15).seconds.do(bot)
bot()

# Run the bot in a loop
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
