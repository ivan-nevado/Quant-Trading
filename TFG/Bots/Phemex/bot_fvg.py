'''This bot will buy when price touches the top of a bullish FVG and sell if it touches the bottom of the bearish FVG
First it will look in at the EMA of 10 periods to see if we are bearish or bullish
If we are bullish it will take bullish FVGs below price but they have to be also above the EMA
If we are bearish it will take bearish FVGs above price but they have to be also below the EMA
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

# Function to calculate the EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=period).mean()

# Function to identify Fair Value Gaps (FVGs)
def identify_fvg(df):
    fvg_zones = []
    for i in range(1, len(df) - 2):
        bottom_bul_fvg = df.High[i]
        top_bul_fvg = df.Low[i+2]
        bottom_bear_fvg = df.Low[i]
        top_bear_fvg = df.High[i+2]

        # Bullish FVG (low of the 1st candle is the bottom, low of the 3rd is the top)
        if bottom_bul_fvg < top_bul_fvg:
            fvg_zones.append((i, 'bullish', bottom_bul_fvg, top_bul_fvg))

        # Bearish FVG (high of the 1st candle is the top, high of the 3rd is the bottom)
        elif bottom_bear_fvg > top_bear_fvg:
            fvg_zones.append((i, 'bearish', bottom_bear_fvg, top_bear_fvg))
    
    return fvg_zones

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
    fvgs = identify_fvg(df)
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

    # Check if there are open orders for this symbol before placing new ones
    if check_open_orders(symbol):
        print(f"Open order exists for {symbol}. Skipping new order.")
        return

    # If not in a position and no open order, apply FVG strategy
    for fvg in fvgs:
        fvg_index, fvg_type, fvg_bottom, fvg_top = fvg

        # Place a BUY order for bullish FVG
        if fvg_type == 'bullish' and current_price > fvg_top and macro_tendency_bullish:
            print(f'Placing BUY order for bullish FVG at {fvg_top}, current price: {current_price}')
            phemex.create_limit_buy_order(symbol, pos_size, fvg_top, params)
            break  # Exit loop after placing the order

        # Place a SELL order for bearish FVG
        elif fvg_type == 'bearish' and current_price < fvg_top and not macro_tendency_bullish:
            print(f'Placing SELL order for bearish FVG at {fvg_top}, current price: {current_price}')
            phemex.create_limit_sell_order(symbol, pos_size, fvg_top, params)
            break  # Exit loop after placing the order

        else:
            print(f'No valid conditions to place an order for FVG type: {fvg_type}, current price: {current_price}, FVG top: {fvg_top}, FVG bottom: {fvg_bottom}')

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

