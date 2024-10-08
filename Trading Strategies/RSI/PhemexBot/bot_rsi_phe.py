'''This bot will look at the ema of 150 periods and see if we are bullish or bearish
If we are bullish and on the rsi we are oversold(below the 30 value) we buy
If we are bearish and on the rsi we are overbought(above the 70 value) we sell'''
import ccxt
import pandas as pd
import numpy as np
import key_file as k  # Ensure this file contains your Phemex API keys
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

# Strategy parameters
ema_period = 150  # EMA period for trend detection
rsi_period = 10  # RSI period for overbought/oversold
overbought = 70  # Overbought threshold for RSI
oversold = 30  # Oversold threshold for RSI

# Function to fetch market data
def fetch_data(symbol, timeframe, limit=1000):
    ohlcv = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']) 
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to calculate the EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=period).mean()

# Function to calculate RSI
def RSI(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to fetch ask and bid prices
def ask_bid():
    try:
        ob = phemex.fetch_order_book(symbol)
        bid = ob['bids'][0][0]
        ask = ob['asks'][0][0]
        return ask, bid
    except ccxt.BaseError as e:
        print(f"Error fetching ask and bid prices: {e}")
        return None, None

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
    if df is None:
        return  # Skip this run if data fetch failed
    
    ask, bid = ask_bid()
    if ask is None or bid is None:
        return  # Skip this run if ask/bid fetch failed

    # Calculate EMA and RSI
    ema = EMA(df['Close'], ema_period)
    df['rsi'] = RSI(df['Close'], rsi_period)
    
    current_price = df['Close'].iloc[-1]
    ema_value = ema.iloc[-1]
    rsi_value = df['rsi'].iloc[-1]
    
    macro_tendency_bullish = current_price > ema_value

    # Check if we have an open position
    open_pos, size, side, entry_price = open_positions()
    in_position = size > 0

    # If in a position, check for exit conditions based on PnL
    if in_position and entry_price is not None:
        pnl_pct = calculate_pnl(entry_price, current_price, side)

        # Exit conditions (example values: target profit 3%, max loss 1%)
        if pnl_pct >= 3 or pnl_pct <= -1:
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

    # Trading logic based on EMA and RSI
    # Buy if bullish trend and RSI indicates oversold
    if macro_tendency_bullish and rsi_value < oversold:
        print(f'Placing BUY order at {current_price}, EMA: {ema_value}, RSI: {rsi_value}')
        try:
            phemex.create_limit_buy_order(symbol, pos_size, ask, params)
        except ccxt.BaseError as e:
            print(f"Error placing BUY order: {e}")

    # Sell if bearish trend and RSI indicates overbought
    elif not macro_tendency_bullish and rsi_value > overbought:
        print(f'Placing SELL order at {current_price}, EMA: {ema_value}, RSI: {rsi_value}')
        try:
            phemex.create_limit_sell_order(symbol, pos_size, bid, params)
        except ccxt.BaseError as e:
            print(f"Error placing SELL order: {e}")
    else:
        print(f'No valid conditions to place an order, current price: {current_price}, ema: {ema_value}, rsi: {rsi_value}')
        
# Schedule the bot to run every 15 seconds
schedule.every(15).seconds.do(bot)

# Initial bot run to ensure the first execution
bot()

# Run the bot in a loop
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(f"Unhandled Error: {e}")
        time.sleep(30)