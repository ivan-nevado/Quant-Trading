'''This bot will buy when price touches the top of a bullish FVG and sell if it touches the bottom of the bearish FVG
First it will look in at the EMA of 10 periods to see if we are bearish or bullish
If we are bullish it will take bullish FVGs below price but they have to be also above the EMA
If we are bearish it will take bearish FVGs above price but they have to be also below the EMA
'''
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

symbol = 'WIF'
pos_size = 10  # Set your position size as a float
target_profit_pct = 8  # Target profit in percentage
max_loss_pct = -9  # Maximum allowed loss in percentage
ema_period = 40  # EMA period, can be adjusted based on optimization results

# Function to fetch market data from Hyperliquid
def fetch_data(symbol, interval='1m', lookback_days=1):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
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
    return ask, bid, l2_data

# Make sure to include the get_ohlcv2 function definition in your script
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

# If not already defined, include process_data_to_df as well
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
        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data


# Function to calculate the EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=period).mean()

# Function to identify Fair Value Gaps (FVGs) using the improved logic
def detect_fvg(df):
    fvg_signals = np.where(
        (
            (df["high"].shift(1) < df["low"].shift(-1))
            & (df["close"] > df["open"])
        )
        | (
            (df["low"].shift(1) > df["high"].shift(-1))
            & (df["close"] < df["open"])
        ),
        np.where(df["close"] > df["open"], 1, -1),
        np.nan,
    )
    
    top = np.where(
        ~np.isnan(fvg_signals),
        np.where(
            df["close"] > df["open"],
            df["low"].shift(-1),
            df["low"].shift(1),
        ),
        np.nan,
    )
    
    bottom = np.where(
        ~np.isnan(fvg_signals),
        np.where(
            df["close"] > df["open"],
            df["high"].shift(1),
            df["high"].shift(-1),
        ),
        np.nan,
    )
    
    # Calculate MitigatedIndex
    mitigated_index = np.zeros(len(df), dtype=np.int32)
    for i in np.where(~np.isnan(fvg_signals))[0]:
        mask = np.zeros(len(df), dtype=np.bool_)
        if fvg_signals[i] == 1:
            mask = df["low"][i + 2:] <= top[i]
        elif fvg_signals[i] == -1:
            mask = df["high"][i + 2:] >= bottom[i]
        if np.any(mask):
            j = np.argmax(mask) + i + 2
            mitigated_index[i] = j

    mitigated_index = np.where(np.isnan(fvg_signals), np.nan, mitigated_index)
    fvg_zones = []
    for i in range(len(fvg_signals)):
        if fvg_signals[i] == 1:
            fvg_zones.append((i, 'bullish', bottom[i], top[i], mitigated_index[i]))
        elif fvg_signals[i] == -1:
            fvg_zones.append((i, 'bearish', bottom[i], top[i], mitigated_index[i]))
    
    return fvg_zones

# Function to determine if an FVG is filled
def is_fvg_filled(fvg, df):
    fvg_index, fvg_type, fvg_bottom, fvg_top, _ = fvg
    for i in range(fvg_index + 2, len(df)):
        if (fvg_type == 'bullish' and df['low'].iloc[i] < fvg_bottom) or \
           (fvg_type == 'bearish' and df['high'].iloc[i] > fvg_top):
            return True  # FVG is filled
    return False  # FVG is not filled

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

def limit_order(coin,is_buy,sz,limit_px, reduced_only, account):
    #account: LocalAccount = eth_account.Account.from_key(key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    #info = Info(constants.MAINNET_API_URL, skip_ws = True)
    #user_state = info.user_state(account.address)
    #print('this is the current account value: {user_state["marginSummary"]["accountValue"]}')

    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz,rounding)
    print(f"coin: {coin}, type: {type(coin)}")
    print(f"is_buy: {is_buy}, type: {type(is_buy)}")
    print(f"sz: {sz}, type: {type(sz)}")
    print(f"limit_px: {limit_px}, type: {type(limit_px)}")
    print(f"reduce_only: {reduced_only}, type: {type(reduced_only)}")


    #limit_px = str(limit_px)
    #sz = str(sz)
    #print(f"limit_px: {limit_px}, type: {type(limit_px)}")
    #print(f"sz: {sz}, type: {type(sz)}")
    print(f'placing limit order for {coin} {sz} @ {limit_px}')
    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=reduced_only)

    if is_buy == True:
        print(f"limit BUY order placed thanks moon, resting: {order_result['response']['data']['statuses'][0]}")
    else:
        print(f"limit SELL order placed thanks moon, resting: {order_result['response']['data']['statuses'][0]}")

    return order_result

# Main bot logic
def bot():
    df = fetch_data(symbol, '1h', 1)
    ask, bid, _ = ask_bid(symbol)
    fvgs = detect_fvg(df)
    current_price = float(df['close'].iloc[-1])
    ema = EMA(df['close'], ema_period)
    ema_value = float(ema.iloc[-1])
    
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
            limit_order(symbol, side == 'Sell', abs(size), ask if side == 'Buy' else bid, True, account)
        else:
            print(f"Holding position with PnL: {pnl_pct}%")
        return

    # Check if there are open orders for this symbol before placing new ones
    if check_open_orders(symbol):
        print(f"Open order exists for {symbol}. Skipping new order.")
        return

    # Reverse the order of FVGs to prioritize the most recent one
    # Reverse the order of FVGs to prioritize the most recent one
    for fvg in reversed(fvgs):
        fvg_index, fvg_type, fvg_bottom, fvg_top, fvg_mit_index = fvg
        
        # Convert fvg_bottom and fvg_top to floats
        fvg_bottom = float(fvg_bottom)
        fvg_top = float(fvg_top)

        # Skip if FVG is already filled or mitigated
        if is_fvg_filled(fvg, df):
            continue

        # Place a BUY order for the most recent bullish FVG
        if fvg_type == 'bullish' and current_price > fvg_top and macro_tendency_bullish and current_price > ema_value:
            print(f'Placing BUY order for bullish FVG at {fvg_top}, current price: {current_price}, EMA: {ema_value}, macro_tendency_bullish: {macro_tendency_bullish}')
            limit_order(symbol, True, pos_size, fvg_top, False, account)
            break  # Exit loop after placing the order

        # Place a SELL order for the most recent bearish FVG
        elif fvg_type == 'bearish' and current_price < fvg_bottom and not macro_tendency_bullish and current_price < ema_value:
            print(f'Placing SELL order for bearish FVG at {fvg_bottom}, current price: {current_price}')
            limit_order(symbol, False, pos_size, fvg_bottom, False, account)
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
