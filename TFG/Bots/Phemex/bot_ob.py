'''This bot will buy and sell in the OrderBlocks
First it will look in the EMA if we are bearish or bullish
If we are bullish it will take bullish OrderBlocks that are above the EMA
If we are bearish it will take bearish  OrderBlocks that are below the EMA
'''
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

# Function to fetch market data
def fetch_data(symbol, timeframe='15m', limit=1000):
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

# Function to detect pivot points (used to detect OBs)
def pivotid(df, l, n1, n2): 
    if l - n1 < 0 or l + n2 >= len(df):
        return 0

    pividlow = True
    pividhigh = True

    for i in range(l - n1, l + 1):
        if df['Low'][l] > df['Low'][i]:
            pividlow = False
    
    for i in range(l + 1, l + n2 + 1):
        if df['Low'][l] > df['Low'][i]:
            pividlow = False
    
    for i in range(l - n1, l + 1):
        if df['High'][l] < df['High'][i]:
            pividhigh = False
    
    for i in range(l + 1, l + n2 + 1):
        if df['High'][l] < df['High'][i]:
            pividhigh = False

    if pividlow:
        return 1  # Pivot Low (Bullish OB Candidate)
    elif pividhigh:
        return 2  # Pivot High (Bearish OB Candidate)
    else:
        return 0

# Function to calculate the EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=period).mean()

# Function to identify OBs
def identify_ob(df):
    ob_zones = []
    for i in range(len(df) - 100, len(df) - 3):  
        pivot = pivotid(df, i, 3, 3)

        if pivot == 1:  
            ob_zones.append((i, 'bullish', df['Low'][i], df['High'][i]))  
        elif pivot == 2:  
            ob_zones.append((i, 'bearish', df['High'][i], df['Low'][i]))

    print(f"Identified OBs: {ob_zones}")
    return ob_zones

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
    df = fetch_data(symbol, '15m', 1000)
    ask, bid = ask_bid()
    obs = identify_ob(df)
    current_price = df['Close'].iloc[-1]
    ema = EMA(df['Close'], 100)
    ema_value = ema.iloc[-1]
    
    macro_tendency_bullish = current_price > ema_value

    # Check if we have an open position
    open_pos, size, side, entry_price = open_positions()
    in_position = size > 0

    # If in a position, check PnL and exit conditions
    if in_position and entry_price is not None:  # Ensure we have an entry price
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

    # Check if there are open orders for this symbol

    # If not in a position and no open order, apply OB strategy
    for ob in obs:
        if check_open_orders(symbol):
            print(f"Open order exists for {symbol}. Skipping new order.")
            return
        ob_index, ob_type, ob_bottom, ob_top = ob
        if ob_type == 'bullish' and current_price > ob_top and macro_tendency_bullish:
            print(f'Placing BUY order for bullish OB at {ob_top}, current price: {current_price}')
            phemex.create_limit_buy_order(symbol, pos_size, ob_top, params)
        elif ob_type == 'bearish' and current_price < ob_top and not macro_tendency_bullish:
            print(f'Placing SELL order for bearish OB at {ob_top}, current price: {current_price}')
            phemex.create_limit_sell_order(symbol, pos_size, ob_top, params)
        else:
            print(f'No valid conditions to place an order for OB type: {ob_type}, current price: {current_price}, OB top: {ob_top}')

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
