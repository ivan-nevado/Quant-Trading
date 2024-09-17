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

symbol = 'WIFUSDT'  # Trading pair
pos_size = 1.0  # Set your position size
params = {'timeInForce': 'PostOnly'}
ema_period = 10  # EMA period for trend detection
rsi_period = 30  # RSI period
overbought = 70  # Overbought threshold for RSI
oversold = 30  # Oversold threshold for RSI

# Function to fetch market data
def fetch_data(symbol, timeframe='1m', limit=1000):
    ohlcv = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']) 
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to calculate EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=period).mean()

# Function to calculate RSI
def RSI(arr, period=14):
    delta = pd.Series(arr).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to fetch ask and bid prices
def ask_bid():
    ob = phemex.fetch_order_book(symbol)
    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    return ask, bid

# Function to check open positions and calculate PnL
def open_positions():
    params = {'type': 'swap', 'code': 'USDT'}
    balance = phemex.fetch_balance(params=params)
    positions = balance['info']['data']['positions']
    
    if len(positions) > 0:
        open_pos = positions[0]  # Assuming a single position
        size = float(open_pos['size'])  # Cast to float
        side = open_pos['side']  # Buy or Sell
        
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
    df = fetch_data(symbol, '1m', 1000)
    ask, bid = ask_bid()
    current_price = df['Close'].iloc[-1]
    ema = EMA(df['Close'], ema_period)
    ema_value = ema.iloc[-1]
    rsi = RSI(df['Close'], rsi_period)
    rsi_value = rsi.iloc[-1]
    
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
    if check_open_orders(symbol):
        print(f"Open order exists for {symbol}. Skipping new order.")
        return

    # Trading logic based on EMA and RSI:
    if macro_tendency_bullish and rsi_value > overbought and not in_position:
        print(f"Placing BUY order as we are bullish and RSI is overbought, price: {current_price}, RSI: {rsi_value}")
        phemex.create_limit_buy_order(symbol, pos_size, bid, params)

    elif not macro_tendency_bullish and rsi_value < oversold and not in_position:
        print(f"Placing SELL order as we are bearish and RSI is oversold, price: {current_price}, RSI: {rsi_value}")
        phemex.create_limit_sell_order(symbol, pos_size, ask, params)

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
