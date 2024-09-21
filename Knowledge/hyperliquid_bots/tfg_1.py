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
account = LocalAccount = eth_account.Account.from_key(secret_key)

symbol = 'WIF'  # Adjust your trading pair
pos_size = 10  # Set your position size
target_profit_pct = 8  # Target profit in percentage
max_loss_pct = -9  # Maximum allowed loss in percentage
ema_period = 100  # EMA period

# Function to fetch market data from Hyperliquid
def fetch_data(symbol, interval, lookback_days=1):
    snapshot_data = get_ohlcv2(symbol, interval, lookback_days)
    df = process_data_to_df(snapshot_data)
    return df

# Fetch order book data (ask and bid prices)
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
    return ask, bid, l2_data

# Fetch OHLCV data for Hyperliquid
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

def process_data_to_df(snapshot_data):
    if snapshot_data:
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        for snapshot in snapshot_data:
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            open_price = float(snapshot['o'])
            high_price = float(snapshot['h'])
            low_price = float(snapshot['l'])
            close_price = float(snapshot['c'])
            volume = float(snapshot['v'])
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=columns)
        # Ensure numeric columns are of type float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        return df
    else:
        return pd.DataFrame()

# Function to calculate EMA
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
            if df1['low'][l] > df1['low'][i]:
                pividlow = False
            if df1['high'][l] < df1['high'][i]:
                pividhigh = False
        if pividlow:
            return 1  # Bullish OB Candidate
        elif pividhigh:
            return -1  # Bearish OB Candidate
        else:
            return 0

    def is_inside_bar(prev_ob_index, new_ob_index, df):
        return df['high'][prev_ob_index] >= df['high'][new_ob_index] and df['low'][prev_ob_index] <= df['low'][new_ob_index]

    def is_pullback(ob_index, mit_index, df, ob_type):
        for j in range(ob_index + 1, mit_index):
            if ob_type == 1 and df['high'][j] < df['high'][j-1]:
                return True
            elif ob_type == -1 and df['low'][j] > df['low'][j-1]:
                return True
        return False

    most_recent_ob_index = None
    for i in range(1, len(df) - 2):
        pivot = pivotid(df, i, 3, 3)

        if pivot == 1:  # Bullish OB
            if most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index, i, df):
                ob_signals[i] = 1
                top[i] = df['high'].iloc[i]
                bottom[i] = df['low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i

        if pivot == -1:  # Bearish OB
            if most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index, i, df):
                ob_signals[i] = -1
                top[i] = df['high'].iloc[i]
                bottom[i] = df['low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i

    for i in np.where(~np.isnan(ob_signals))[0]:
        ob_type = ob_signals[i]
        if ob_type == 1:  # Bullish OB
            for j in range(i + 1, len(df)):
                if df['low'].iloc[j] <= top[i]:
                    if is_pullback(i, j, df, ob_type):
                        mitigated_index[i] = j
                        break
        elif ob_type == -1:  # Bearish OB
            for j in range(i + 1, len(df)):
                if df['high'].iloc[j] >= bottom[i]:
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

# Check for open positions and calculate PnL
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
    '''
    this is successfully returns Size decimals and Price decimals

    this outputs the size decimals for a given symbol
    which is - the SIZE you can buy or sell at
    ex. if the sz decimal == 1 then you can buy/sell 1.4
    if sz decimal == 2 then you can buy/sell 1.45
    if sz decimal == 3 then you can buy/sell 1.456

    if size isnt right, we get this error, to avoid it use the sz decimal func
    {'error': 'Invalid order size'}
    '''

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, data = json.dumps(data))

    if response.status_code == 200:
        #Success
        data = response.json()
        #print(data)
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
        else:
            print('Symbol not found')

    else:
        #Error
        print('Error', response.status_code)

    ask = ask_bid(symbol)[0]
    #print(f"this is for the ask {ask}")

    #Compute the numbers of decimal points in the ask price
    ask_str = str(ask)
    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0

    print(f'{symbol} this is the price {sz_decimals} decimals(s)')

    return sz_decimals, px_decimals

# Place a limit order
def limit_order(coin, is_buy, sz, limit_px, reduced_only, account):
    exchange = Exchange(account, constants.MAINNET_API_URL)
    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz, rounding)

    print(f'Placing limit order for {coin}: {sz} @ {limit_px}')
    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=reduced_only)

    if is_buy:
        print(f"Limit BUY order placed: {order_result['response']['data']['statuses'][0]}")
    else:
        print(f"Limit SELL order placed: {order_result['response']['data']['statuses'][0]}")

    return order_result

# Main bot logic
def bot():
    df = fetch_data(symbol, '1h', 1)
    ask, bid, _ = ask_bid(symbol)
    obs = identify_ob(df)
    current_price = float(df['close'].iloc[-1])
    ema = EMA(df['close'], ema_period)
    ema_value = float(ema.iloc[-1])

    macro_trend_bullish = current_price > ema_value

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

    # Loop through identified OBs in reverse order
    for ob in reversed(list(obs.itertuples())):
        ob_type, ob_top, ob_bottom, ob_mit_index = ob.OB, ob.Top, ob.Bottom, ob.MitigatedIndex
        print(f"OB Type: {ob.OB}, OB Top: {ob.Top}, OB Bottom: {ob.Bottom}, Current Price: {current_price}, ema_value: {ema_value}")

        # Place a BUY order for the most recent bullish OB
        if ob_type == 1:
            if current_price > ob_top:
                if macro_trend_bullish:
                    if current_price > ema_value:
                        print(f'Placing BUY order for bullish OB at {ob_top}, current price: {current_price}, EMA: {ema_value}')
                        limit_order(symbol, True, pos_size, ob_top, False, account)
                        break

        # Place a SELL order for the most recent bearish OB
        elif ob_type == -1 and current_price < ob_bottom and not macro_trend_bullish and current_price < ema_value:
            print(f'Placing SELL order for bearish OB at {ob_bottom}, current price: {current_price}')
            limit_order(symbol, False, pos_size, ob_bottom, False, account)
            break
        else:
            print('No conditions met to place an order')

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
