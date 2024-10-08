import dontshare as d
import eth_account
from eth_account.signers.local import LocalAccount
import json
import time
import pandas as pd
import numpy as np
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import schedule
import requests
from datetime import datetime, timedelta

# Initialize Hyperliquid client with API keys
secret_key = d.private_key
account = eth_account.Account.from_key(secret_key)

symbol = 'WIF'
pos_size = 10  # Set your position size as a float
target_profit_pct = 3  # Target profit in percentage (adjust as needed)
max_loss_pct = -1  # Maximum allowed loss in percentage (adjust as needed)
ema_period = 150  # EMA period for trend detection
rsi_period = 10  # RSI period for overbought/oversold
overbought = 70  # Overbought threshold for RSI
oversold = 30  # Oversold threshold for RSI

# Function to fetch market data from Hyperliquid
def fetch_data(symbol, interval, lookback_days=1):
    snapshot_data = get_ohlcv2(symbol, interval, lookback_days)
    df = process_data_to_df(snapshot_data)
    return df

# Function to fetch ask and bid prices from Hyperliquid
def ask_bid(symbol):
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        'type': 'l2Book',
        'coin': symbol
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])
    return ask, bid

# Function to get OHLCV data from Hyperliquid
def get_ohlcv2(symbol, interval, lookback_days):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000)
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        snapshot_data = response.json()
        return snapshot_data
    else:
        print(f'Error fetching data for {symbol}: {response.status_code}')
        return None

# Function to process OHLCV data into DataFrame
def process_data_to_df(snapshot_data):
    if snapshot_data:
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        for snapshot in snapshot_data:
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            open_price = snapshot['o']
            high_price = snapshot['h']
            low_price = snapshot['l']
            close_price = snapshot['c']
            volume = snapshot['v']
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=columns)
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data

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

# Function to check open positions and calculate PnL
def open_positions():
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    positions = user_state["assetPositions"]
    
    if len(positions) > 0:
        for position in positions:
            if float(position["position"]["szi"]) != 0:
                size = float(position["position"]["szi"])
                side = 'Buy' if size > 0 else 'Sell'
                entry_px = float(position["position"]["entryPx"])
                return position, size, side, entry_px
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
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    open_orders = info.open_orders(account.address)
    return len(open_orders) > 0  # Returns True if there are open orders, False otherwise

def get_sz_px_decimals(symbol):
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
        else:
            print('Symbol not found')
            sz_decimals = 0
    else:
        print('Error', response.status_code)
        sz_decimals = 0

    ask = ask_bid(symbol)[0]

    # Compute the numbers of decimal points in the ask price
    ask_str = str(ask)
    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0

    return sz_decimals, px_decimals

def limit_order(coin,is_buy,sz,limit_px, reduced_only, account):
    exchange = Exchange(account, constants.MAINNET_API_URL)
    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz, rounding)

    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=reduced_only)

    if is_buy:
        print(f"Limit BUY order placed, status: {order_result['response']['data']['statuses'][0]}")
    else:
        print(f"Limit SELL order placed, status: {order_result['response']['data']['statuses'][0]}")

    return order_result

# Main bot logic
def bot():
    df = fetch_data(symbol, '1h', 1)  # Adjust interval as needed
    ask, bid = ask_bid(symbol)
    
    # Calculate EMA and RSI
    ema = EMA(df['close'], ema_period)
    df['rsi'] = RSI(df['close'], rsi_period)
    
    current_price = float(df['close'].iloc[-1])
    ema_value = float(ema.iloc[-1])
    rsi_value = float(df['rsi'].iloc[-1])
    
    macro_tendency_bullish = current_price > ema_value

    # Check if we have an open position
    open_pos, size, side, entry_price = open_positions()
    in_position = size > 0

    # If in a position, check PnL and exit conditions
    if in_position and entry_price is not None:
        pnl_pct = calculate_pnl(entry_price, current_price, side)
        print(f"Current PnL: {pnl_pct}%")

        if pnl_pct >= target_profit_pct or pnl_pct <= max_loss_pct:
            print(f"Exiting position at {current_price} with PnL: {pnl_pct}%")
            limit_order(symbol, side == 'Sell', abs(size), ask if side == 'Buy' else bid, True, account)
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
        limit_order(symbol, True, pos_size, ask, False, account)

    # Sell if bearish trend and RSI indicates overbought
    elif not macro_tendency_bullish and rsi_value > overbought:
        print(f'Placing SELL order at {current_price}, EMA: {ema_value}, RSI: {rsi_value}')
        limit_order(symbol, False, pos_size, bid, False, account)
    else:
        print(f'No valid conditions to place an order, current price: {current_price}, EMA: {ema_value}, RSI: {rsi_value}')

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
