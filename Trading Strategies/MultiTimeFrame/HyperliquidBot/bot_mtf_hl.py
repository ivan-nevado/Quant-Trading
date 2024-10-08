import eth_account
import json
import time
import pandas as pd
import numpy as np
import schedule
import requests
from datetime import datetime, timedelta
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import dontshare as d

# Initialize Hyperliquid client with API keys
secret_key = d.private_key
account = eth_account.Account.from_key(secret_key)

symbol = 'WIF'  # Adjust your trading pair
pos_size = 10  # Set your position size
target_profit_pct = 8  # Target profit in percentage
max_loss_pct = -9  # Maximum allowed loss in percentage
ema_period = 100  # EMA period
sl_offset = 0.001  # Stop-loss offset
risk_reward_ratio = 3  # Risk/Reward ratio

# Function to fetch market data from Hyperliquid
def fetch_data(symbol, interval, lookback_days=1):
    snapshot_data = get_ohlcv2(symbol, interval, lookback_days)
    df = process_data_to_df(snapshot_data)
    return df

# Fetch order book data (ask and bid prices)
def ask_bid(symbol):
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'l2Book', 'coin': symbol}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()['levels']
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])
    return ask, bid

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
    return response.json() if response.status_code == 200 else None

def process_data_to_df(snapshot_data):
    if snapshot_data:
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = [
            [
                datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                float(snapshot['o']), float(snapshot['h']),
                float(snapshot['l']), float(snapshot['c']), float(snapshot['v'])
            ]
            for snapshot in snapshot_data
        ]
        df = pd.DataFrame(data, columns=columns)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    return pd.DataFrame()

# Function to calculate EMA
def EMA(series, period):
    return pd.Series(series).ewm(span=period, min_periods=1).mean()

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
            if df1['low'].iloc[l] > df1['low'].iloc[i]:
                pividlow = False
            if df1['high'].iloc[l] < df1['high'].iloc[i]:
                pividhigh = False
                
        if pividlow:
            if (df1["high"].iloc[l] < df1["low"].iloc[l+2]) and (df1["close"].iloc[l+1] > df1["open"].iloc[l+1]):
                return 1  # Pivot Low (Bullish OB Candidate)
        elif pividhigh:
            if (df1["low"].iloc[l] > df1["high"].iloc[l+2]) and (df1["close"].iloc[l+1] < df1["open"].iloc[l+1]):
                return -1  # Pivot High (Bearish OB Candidate)
        return 0

    def is_inside_bar(prev_ob_index, new_ob_index, df):
        return df['high'].iloc[prev_ob_index] >= df['high'].iloc[new_ob_index] and df['low'].iloc[prev_ob_index] <= df['low'].iloc[new_ob_index]

    def is_pullback(ob_index, mit_index, df, ob_type):
        inside_bar_check = True
        start_index = ob_index + 2
        for j in range(start_index, mit_index + 1):
            if ob_type == 1:  # Bullish OB
                if inside_bar_check:
                    inside_bar = is_inside_bar(ob_index, j, df)
                    if not inside_bar:
                        inside_bar_check = False
                if not inside_bar_check and df['high'].iloc[j] < df['high'].iloc[j - 1]:
                    return True
            elif ob_type == -1:  # Bearish OB
                if inside_bar_check:
                    inside_bar = is_inside_bar(ob_index, j, df)
                    if not inside_bar:
                        inside_bar_check = False
                if not inside_bar_check and df['low'].iloc[j] > df['low'].iloc[j - 1]:
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
                    mitigated_index[i] = j
                    break
        elif ob_type == -1:  # Bearish OB
            for j in range(i + 1, len(df)):
                if df['high'].iloc[j] >= bottom[i]:
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
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    symbols = response.json()['universe'] if response.status_code == 200 else None
    symbol_info = next((s for s in symbols if s['name'] == symbol), None) if symbols else None
    sz_decimals = symbol_info['szDecimals'] if symbol_info else None

    ask_str = str(ask_bid(symbol)[0])
    px_decimals = len(ask_str.split('.')[1]) if '.' in ask_str else 0

    return sz_decimals, px_decimals

# Place a limit order
def limit_order(coin, is_buy, sz, limit_px, reduced_only, account):
    exchange = Exchange(account, constants.MAINNET_API_URL)
    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz, rounding)
    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=reduced_only)
    return order_result

# Main bot logic
def bot():
    df_HTF = fetch_data(symbol, '1h', 1000)  # Fetch 1h timeframe data
    df_LTF = fetch_data(symbol, '15m', 1000)  # Fetch 15m timeframe data
    ask, bid = ask_bid(symbol)
    obs_HTF = identify_ob(df_HTF)
    obs_LTF = identify_ob(df_LTF)

    current_price = float(df_HTF['close'].iloc[-1])
    ema_HTF = EMA(df_HTF['close'], ema_period).iloc[-1]

    macro_trend_bullish = current_price > ema_HTF

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

    # Now iterate through the HTF OBs and check for mitigation
    for ob_HTF in reversed(list(obs_HTF.itertuples())):
        if pd.isna(ob_HTF.OB):
            print(f"No valid order block found: {ob_HTF}")
            continue
        ob_type_HTF, ob_top_HTF, ob_bottom_HTF, ob_mit_index_HTF, ob_index_HTF = ob_HTF.OB, ob_HTF.Top, ob_HTF.Bottom, ob_HTF.MitigatedIndex, ob_HTF.OBIndex
        current_index = df_HTF.index[-1]

        if current_index == ob_mit_index_HTF:  # If HTF OB is mitigated
            if macro_trend_bullish:
                if ob_type_HTF == 1 and current_price < ob_top_HTF:  # Bullish OB
                    print(f"Bullish HTF OB mitigated at :{ob_HTF}, checking LTF OB")
                    for ob_LTF in reversed(list(obs_LTF.itertuples())):
                        ob_type_LTF, ob_top_LTF, ob_bottom_LTF = ob_LTF.OB, ob_LTF.Top, ob_LTF.Bottom

                        if ob_type_LTF == 1 and current_price > ob_top_LTF:  # Bullish LTF OB
                            print(f"Placing BUY order for bullish OB at {ob_top_LTF}, current price: {current_price}")
                            limit_order(symbol, True, pos_size, ob_LTF.Top, False, account)
                            return  # Exit after placing the order
            elif not macro_trend_bullish:
                if ob_type_HTF == -1 and current_price > ob_bottom_HTF and ob_bottom_LTF >= ob_bottom_HTF and ob_top_LTF<= ob_top_HTF:  # Bearish OB
                    print(f"Bearish HTF OB mitigated at :{ob_HTF}, checking LTF OB")
                    for ob_LTF in reversed(list(obs_LTF.itertuples())):
                        ob_type_LTF, ob_top_LTF, ob_bottom_LTF = ob_LTF.OB, ob_LTF.Top, ob_LTF.Bottom

                        if ob_type_LTF == -1 and current_price < ob_bottom_LTF and ob_bottom_LTF >= ob_bottom_HTF and ob_top_LTF<= ob_top_HTF:  # Bearish LTF OB
                            print(f"Placing SELL order for bearish OB at {ob_bottom_LTF}, current price: {current_price}")
                            limit_order(symbol, False, pos_size, ob_LTF.Bottom, False, account)
                            return  # Exit after placing the order
        else:
            print(f'Havent mitigated the HTF ob: {ob_HTF}')
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
