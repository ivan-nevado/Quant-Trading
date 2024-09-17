'''This bot will look for the macro tendency first,knowing if it is bullish or bearish basing on the MS code of the library 
then if we are in a bearish tendency it will look in the support and resistance indicator,
for breaks of supports and in a bullish range for breaks of resistance
it will buy after a break of resistance and sell after a break of a support'''

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
buffer_pct = 0.1  # 10% buffer for support/resistance

# Function to fetch market data
def fetch_data(symbol, timeframe='1m', limit=1000):
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

# Support and resistance detection (as in backtesting)
def is_support(df, l, n1, n2):
    for i in range(l - n1 + 1, l + 1):
        if float(df.Low[i]) > float(df.Low[i - 1]):
            return False
    for i in range(l + 1, l + n2 + 1):
        if float(df.Low[i]) < float(df.Low[i - 1]):
            return False
    return True

def is_resistance(df, l, n1, n2):
    for i in range(l - n1 + 1, l + 1):
        if float(df.High[i]) < float(df.High[i - 1]):
            return False
    for i in range(l + 1, l + n2 + 1):
        if float(df.High[i]) > float(df.High[i - 1]):
            return False
    return True

def calculate_support_resistance(df, n1=3, n2=3):
    supports = np.zeros(len(df))
    resistances = np.zeros(len(df))

    # Only loop over the range where there's enough data on both sides
    for row in range(n1, len(df) - n2):  # Skip the last `n2` candles
        if is_support(df, row, n1, n2):
            supports[row] = float(df.Low[row])
        if is_resistance(df, row, n1, n2):
            resistances[row] = float(df.High[row])

    return supports, resistances


# Function to calculate the EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=period).mean()

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

def bot():
    df = fetch_data(symbol, '1m', 1000)
    ask, bid = ask_bid()
    current_price = df['Close'].iloc[-1]
    ema = EMA(df['Close'], 100)
    ema_value = ema.iloc[-1]
    
    macro_tendency_bullish = current_price > ema_value
    
    # Calculate support and resistance levels
    supports, resistances = calculate_support_resistance(df)

    # Get the most recent non-zero support and resistance values
    support = supports[supports > 0][-1] if np.any(supports > 0) else None
    resistance = resistances[resistances > 0][-1] if np.any(resistances > 0) else None

    # Print out support and resistance for debugging
    print(f"Most recent support: {support}, resistance: {resistance}, current price: {current_price}")

    # Add a check to make sure you only proceed if support and resistance are not None
    if support:
        support_buffer = support * (1 + buffer_pct)
    if resistance:
        resistance_buffer = resistance * (1 - buffer_pct)

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

    # Apply the Order Block (OB) strategy
    if support and support <= current_price <= support_buffer and macro_tendency_bullish:
        print(f'Placing BUY order for bullish support at {bid}, support: {support}')
        phemex.create_limit_buy_order(symbol, pos_size, bid, params)
    elif resistance and resistance_buffer <= current_price <= resistance and not macro_tendency_bullish:
        print(f'Placing SELL order for bearish resistance at {ask}, resistance: {resistance}')
        phemex.create_limit_sell_order(symbol, pos_size, ask, params)
    else:
        print(f'No valid conditions to place an order for support: {support}, resistance: {resistance}, current price: {current_price}')

    
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
