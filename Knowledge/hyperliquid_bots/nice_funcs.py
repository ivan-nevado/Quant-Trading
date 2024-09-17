import dontshare as d
import nice_funcs as n
from eth_account.signers.local import LocalAccount
import eth_account
import json,time
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import ccxt
import pandas as pd
import datetime
from datetime import datetime, timedelta
import schedule
import requests
import pandas_ta as ta

secret_key = d.private_key

symbol = 'WIF'

def get_position_andmaxpos(symbol, account, max_positions):
    '''
    gets the current position info, like size etc
    '''

    #account = LocalAccount = eth_account.Account.from_key(secret_key)
    info = Info(constants.MAINNET_API_URL, skip_ws= True)
    user_state = info.user_state(account.address)
    print(f'this is the current account value: {user_state["marginSummary"]["accountValue"]}')
    positions = []
    open_positions = []
    #print(f'this is the symbol: {symbol}')
    #print(user_state["assetPositions"])

    #CHECKING MAX POSITIONS FIRST
    #Iterate over each position in the assetPositions list
    for position in user_state["assetPositions"]:
        #Check if the position size ('szi') is not zero, indicating an open position
        if float(position["position"]["szi"]) != 0:
            #If it's an open position, add the coin symbol to the open positions list
            open_positions.append(position["position"]["coin"])

    #print(open_positions)
    num_of_pos = len(open_positions)
    print(f'we are in {len(positions)} positions and max pos is {max_positions}...closing positions')

    if len(open_positions) > max_positions:

        print(f'we are in {len(open_positions)} positions and max pos is {max_positions}... closing positions')

        #for position in positions we need to call the kill switch
        for position in open_positions:
            kill_switch(position,account)

    else:
        print(f'we are in {len(open_positions)} positions and max pos is {max_positions}... not closing positions')

    for position in user_state["assetPositions"]:
        if(position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            print(f'this is the pnl perc {pnl_perc}')
            break

    else:
        in_pos = False
        size = 0
        pos_sym = None
        entry_px = 0
        pnl_perc = 0

    if size > 0:
        long = True
    elif size < 0:
        long = False
    else:
        long = None

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long, num_of_pos



def get_position(symbol, account):
    '''gets the position info we need'''
    info = Info(constants.MAINNET_API_URL, skip_ws = True)
    user_state = info.user_state(account.address)
    print(f'this is the current account val {user_state["marginSummary"]["accountValue"]}')
    positions = []
    print(f'this is the symbol {symbol}')
    for position in user_state["assetPositions"]:
        if(position["position"]["coin"] == symbol)and float(position["position"]["szi"])!=0:
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]['coin']
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"])*100
            print(f'this is the pnl perc {pnl_perc}')
            break
    else:
        in_pos = False
        size = 0
        pos_sym = None
        entry_px = 0
        pnl_perc = 0

    if size > 0:
        long = True
    elif size < 0:
        long = False
    else:
        long = None

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long


def ask_bid(symbol):
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {
        'type': 'l2Book',
        'coin': symbol
    }

    response = requests.post(url, headers=headers, data = json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']
    #print(l2_data)

    #get bid and ask
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data

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

def cancell_all_orders(account):
    #this cancells all open orders
    #account = LocalAccount = eth_account.Account.from_key(secret_key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws = True)

    open_orders = info.open_orders(account.address)
    #print(open_orders)

    print('above are the open orders... need to cancel any...')
    for open_order in open_orders:
        #print(f'cancelling order {open_order}')
        exchange.cancel(open_order['coin'], open_order['oid'])


def kill_switch(symbol, account):
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
    while im_in_pos:
        cancell_all_orders(account)

        # get bid ask
        ask,bid,l2_data = ask_bid(pos_sym)

        pos_size = abs(pos_size)

        if long == True:
            limit_order(pos_sym, False, pos_size, ask,True,account)
            print('kill switch sell to close submitted')
            time.sleep(5)
        elif long == False:
            limit_order(pos_sym,True,pos_size, bid,True,account)
            print('kill switch to buy to close submitted')
            time.sleep(5)

        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    print('position successfully closed in kill switch')


def pnl_close(symbol,target, max_loss, account):
    #this checks if we hit our target or max loss

    print('entering pnl close')
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    if pnl_perc > target:
        print(f'pnl gain is {pnl_perc} and target is {target} closing position as a WIN')
        kill_switch(pos_sym, account)
    elif pnl_perc <= max_loss:
        print(f'pnl loss is {pnl_perc} and max loss is {max_loss} closing position as a LOSS')
        kill_switch(pos_sym, account)
    else:
        print(f'pnl loss is {pnl_perc} and max loss is {max_loss} target {target} not closing')

    print('finished pnl close')

def acct_bal(account):

    account: LocalAccount = eth_account.Account.from_key(secret_key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws = True)
    user_state = info.user_state(account.address)
    print(f'this is the current account value: {user_state["marginSummary"]["accountValue"]}')

    value = user_state["marginSummary"]["accountValue"]

    return value

def adjust_leverage_size_signal(symbol, leverage, account):
    '''
    this calculates size based off what we want
    95% of balance
    '''

    print('leverage:', leverage)

    # account = LocalAccount = eth_account.Account.from_key(secret_key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws= True)

    # Get the user state and print out leverage information for ETH
    user_state = info.user_state(account.address)
    acct_value = user_state["marginSummary"]["accountValue"]
    acct_value = float(acct_value)

    acct_val95 = acct_value * .95

    print(exchange.update_leverage(leverage, symbol))

    price = ask_bid(symbol)[0]

    #size == balance / price * leverage
    #INJ 6.95 .. at 10x Lev.. 10 INJ == $cost 6.95
    size = (acct_val95/price) * leverage
    size = float(size)
    rounding = get_sz_px_decimals(symbol)[0]
    size = round(size, rounding)
    # print(f'this is the size we can use 95% of acct val {size}')

    user_state = info.user_state(account.address)

    return leverage,size


def get_ohlcv2(symbol, interval, lookback_days):
    end_time = datetime.now()
    start_time = end_time - timedelta(days = lookback_days)

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin":symbol,
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
        #Assuming the response contains a list of candles
        columns = ['timestamp','open','high','low','close','volume']
        data = []
        for snapshot in snapshot_data:
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            open_price = snapshot['o']
            high_price = snapshot['h']
            low_price = snapshot['l']
            close_price = snapshot['c']
            volume = snapshot['v']
            data.append([timestamp, open_price, high_price,low_price,close_price,volume])

        df = pd.DataFrame(data, columns=columns)

        #Calculate support and resistance, excluding the last two rows for the calculation
        if len(df) > 2: #Checks if Dataframe has more than 2 rows to avoid errors
            df['support'] = df[:-2]['close'].min()
            df['resis'] = df[:-2]['close'].max()
        else:
            df['support'] = df['close'].min()
            df['resis'] = df['close'].max()
        
        return df
    else:
        return pd.DataFrame() #Return empty Dataframe if no data

def calculate_bollinger_bands(df, length = 20, std_dev = 2):
    '''
    Calculate Bolling Bands for a given Dataframe and classify when the bands are tight vs wide.

    Parameters:
    -df: Dataframe with a 'close' column.
    -Length: The period over which the SMA is calculated. Default is 20.
    -std_dev: The number of standard deviations to plot above and below the SMA. Default is 2.

    Returns:
    -df: Dataframe with Bollinger Bands and classification for 'tight' and 'wide' bands.
    -tight: Boolean indicating if the bands are currently tight.
    -wide: Boolean indicating if the bands are currently wide.
    '''

    #Ensure 'close' is numeric
    df['close'] = pd.to_numeric(df['close'], errors = 'coerce')
    
    #Calculate Bollinger Bands using pandas_ta
    bollinger_bands = ta.bbands(df['close'], length=length, std=std_dev)

    #Select only the Bollinger Bands columns (ignoring additional columns like bandwidth and percent bandwith)
    bollinger_bands = bollinger_bands.iloc[:, [0,1,2]] #Assuming the first 3 columns are BBL, BBM, and BBU
    bollinger_bands.columns = ['BBL', 'BBM', 'BBU']

    #Merge the Bollinger Bands into the original Dataframe
    df = pd.concat([df, bollinger_bands], axis = 1)

    #Calculate Band Width
    df['BandWidth'] = df['BBU'] - df['BBL']

    #Determine thresholds for 'tight' and 'wide' bands
    tigh_threshold = df['BandWidth'].quantile(0.2)
    wide_threshold = df['BandWidth'].quantile(0.8)

    #Classify the current state of the bands
    current_band_width = df['BandWidth'].iloc[-1]
    tight = current_band_width <= tigh_threshold
    wide = current_band_width >= wide_threshold

    return df, tight, wide

def close_all_positions(account):

    info = Info(constants.MAINNET_API_URL,skip_ws= True)
    user_state = info.user_state(account.address)
    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    positions = []
    open_positions = []
    print(f'this is the symbol {symbol}')
    print(user_state["assetPositions"])

    #cancel all orders
    cancell_all_orders(account)
    print('all orders have been cancelled')

    #CHECKING MAX POSITIONS FIRST
    #Iterate over each positions in the assetPositions list
    for position in user_state["assetPositions"]:
        #Check if the position size ('szi') is not zero, indicating an open position
        if float(position["position"]["szi"])!=0:
            #If it's an open position, add the coin symbol to the open_positions list
            open_positions.append(position["position"]["coin"])

    #for position in positions we need to call the kill switch
    for position in open_positions:
        kill_switch(position,account)

    print('all positions have been closed')

def fetch_candle_snapshot(symbol,interval,start_time,end_time):
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type':'application/json'}
    data = {
        "type":"candleSnapshot",
        "req": {
            "coin":symbol,
            "interval":interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000)
        }
    }

    response = requests.post(url,headers=headers, json = data)
    if response.status_code == 200:
        snapshot_data = response.json()
        return snapshot_data
    else:
        print(f'Error fetching data for {symbol}: {response.status_code}')
        return None

def calculate_sma(prices, window):
    sma = prices.rolling(window=window).mean()
    return sma.iloc[-1]

def get_latest_sma(symbol, interval, window, lookback_days):

    start_time = datetime.now() - timedelta(days = lookback_days)
    end_time = datetime.now()

    snapshots = fetch_candle_snapshot(symbol, interval, start_time, end_time)

    if snapshots:
        prices = pd.Series([float(snapshot['c']) for snapshot in snapshots])
        latest_sma = calculate_sma(prices,window)
        return latest_sma
    else:
        return None
    
def supply_demand_zones_hl(symbol, timeframe, limit):
    print('Starting supply and demand zone calculations...')
    
    # Initialize an empty DataFrame
    sd_df = pd.DataFrame()
    
    # Fetch and process data
    snapshot_data = get_ohlcv2(symbol, timeframe, limit)
    df = process_data_to_df(snapshot_data)
    
    # Calculate support and resistance levels
    df['supp_lo'] = df['low'].rolling(window=2).min()  # Last two periods low for support
    df['res_hi'] = df['high'].rolling(window=2).max()  # Last two periods high for resistance
    
    # Get the last values
    supp_lo = df.iloc[-1]['supp_lo']
    res_hi = df.iloc[-1]['res_hi']
    
    # Assuming 'support' and 'resis' are calculated elsewhere or placeholders
    supp = supp_lo  # You might want to replace this with the actual 'support' logic
    resis = res_hi  # You might want to replace this with the actual 'resis' logic
    
    # Populate the supply and demand zones DataFrame
    sd_df[f'{timeframe}_dz'] = [supp_lo, supp]
    sd_df[f'{timeframe}_sz'] = [res_hi, resis]
    
    # Print the supply and demand zones
    print('Here are the supply and demand zones:')
    print(sd_df)
    
    return sd_df

def calculate_vwap_with_symbol(symbol):
    #fetch and process data
    snapshot_data = get_ohlcv2(symbol, '15m', 300)
    df = process_data_to_df(snapshot_data)

    #Converts the 'timestamp' column to datetime and set that as an index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace = True)

    #Ensure all columns used for VWAP calculation are of numeric type
    numeric_columns = ['high', 'low', 'close', 'volume']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors = 'coerce') # coerce will set errors to NaN

    #Drop rows with NaNs created during type conversion (if any)
    df.dropna(subset=numeric_columns, inplace = True)

    #Ensure the DataFrame is ordered by datetime
    df.sort_index(inplace = True)

    #Calculate VWAP and add it as a new column
    df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close = df['close'], volume=df['volume'])

    #Retrieve the latest VWAP value from the DataFrame
    latest_vwap = df['VWAP'].iloc[-1]

    return df, latest_vwap
                                        