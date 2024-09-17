import ccxt
import pandas as pd
import numpy as np
import key_file as k
from datetime import date,datetime, timezone, tzinfo
import time, schedule

phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': k.key,
    'secret': k.secret
})

symbol = 'WIFUSDT'
pos_size = 1
params = {'timeInForce': 'PostOnly'}
target = 8
max_loss = -9
vol_decimal = .4
timeframe = '15m'
limit = 100
sma = 20

# open positions
def open_positions(symbol=symbol):

    #what is the position index for that symbol?
    if symbol == 'BTCUSD':
        index_pos = 4
    elif symbol == 'APEUSD':
        index_pos = 2
    elif symbol == 'ETHUSDT':
        index_pos = 3
    elif symbol == 'DOGEUSD':
        index_pos = 1
    elif symbol == 'WIFUSDT':
        index_pos = 0
    else:
        index_pos = None
        
    params = {'type':'swap','code':'USDT'}
    phe_bal = phemex.fetch_balance(params=params)
    open_positions = phe_bal['info']['data']['positions']
    #print(open_positions)

#dictionaries
    open_pos_side = open_positions[index_pos]['side'] #btc [3][0] = doge, [1] ape
    open_pos_size = open_positions[index_pos]['size']
    #print(open_positions)

#if statements
    if open_pos_side == ('Buy'):
        openpos_bool = True
        long = True
    elif open_pos_side == ('Sell'):
        openpos_bool = True
        long = False
    else:
        openpos_bool = False
        long = None

        print(f'open_positions... | openpos_bool {openpos_bool} | openpos_size {open_pos_size} | long {long} | index_pos {index_pos}')

#returning
    return open_positions, openpos_bool, open_pos_size, long, index_pos

#ask_bid
def ask_bid(symbol=symbol):
    ob = phemex.fetch_order_book(symbol)
    #print(ob)

    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]

#f literal
    print(f'this is the ask for {symbol} {ask}')

    return ask, bid #ask_bid()[0] = ask, [1] = bid


# pnl close
# pnl close() [0] pnl close and [1] in_pos [2] size [3] long TF
def pnl_close(symbol=symbol, target=target,max_loss = max_loss):
    
    print(f'checking to see if its time to exit for {symbol}...')

    params = {'type':"swap", 'code':'USD'}
    pos_dict = phemex.fetch_positions(params=params)
    #print(pos_dict)

    index_pos = open_positions(symbol)[4]
    pos_dict = pos_dict[index_pos] #btc [3] [0] = doge, [1] = ape
    side = pos_dict['side']
    size = pos_dict['contracts']
    entry_price = float(pos_dict['entryPrice'])
    leverage = float(pos_dict['leverage'])

    current_price = ask_bid(symbol)[1]

    print(f'side: {side} | entry_price: {entry_price} | lev: {leverage}')
    #short or long

    if side == 'long':
        diff = current_price - entry_price
        long = True
    else:
        diff = entry_price - current_price
        long = False

#try / except
    try:
        perc = round(((diff/entry_price) * leverage), 10)
    except:
        perc = 0

    perc = 100*perc
    print(f'for {symbol} this is our PNL percentage: {(perc)}%')

    pnlclose = False
    in_pos = False

    if perc > 0:
        in_pos = True
        print(f'for {symbol} we are in a winning position')
        if perc > target:
            print(':) :) we are in profit & hit target.. checking volume to see if we should start kill switch')
            pnl_close = True
            kill_switch(symbol)
        else:
            print('we have not hit our target yet')

    elif perc < 0: # -10, -20,

        in_pos = True

        if perc <= max_loss: #under -55, -5
            print(f'we need to exit now down {perc}... so starting the kill switch.. max loss {max_loss}')
            kill_switch(symbol)
        else:
            print(f'we are in a losing position of {perc}.. but chilling cause max loss is {max_loss}')

    else:
        print('we are not in a position')

    print(f'for {symbol} just finished checking the PNL close..')

    return pnlclose, in_pos, size, long

def sleep_on_close():
    '''
    this func pulls closed orders, then if last close was in last 59min
    then it sleeps for 1m
    sincelasttrade = minutes since last trade
    '''

    closed_orders = phemex.fetch_closed_orders(symbol)
    #print(closed_orders)

    for ord in closed_orders[-1::-1]:

        sincelasttrade = 59 # how long we pause

        filled = False

        status = ord['info']['ordStatus']
        txttime = ord['info']['transactTimeNs']
        txttime = int(txttime)
        txttime = round((txttime/1000000000)) #bc in nanoseconds
        print(f'this is the status of the order {status} with epoch {txttime}')
        print('next iteration...')
        print('-----')

        if status == 'Filled':
            print('FOUND the order with last fill...')
            print(f'this is the time {txttime} this si the orderstatus {status}')
            orderbook = phemex.fetch_order_book(symbol)
            ex_timestamp = int(ex_timestamp/1000)
            print(txttime)
            print(ex_timestamp)

            time_spread = (ex_timestamp - txttime)/60

            if time_spread < sincelasttrade:
                #print('time since last trade is less than time spread')
                #if in pos is true, put a close order here
                #if in_pos == True:

                sleepy = round(sincelasttrade-time_spread) * 60
                sleepy_min = sleepy/60

                print(f'the time speed is less than {sincelasttrade} mins its been {time_spread}mins... so we SLEEPING for 60secs...')
                time.sleep(60)

            else:
                print(f'its been {time_spread}mins since last fill not sleeping cuz since last trade is {sincelasttrade}')
            break
        else:
            continue

    print('done with sleep on close function..')

# kill switch
def kill_switch(symbol=symbol):
    
    print(f'starting the kill switch for {symbol}')
    openposi = open_positions(symbol)[1] # true or false
    long = open_positions(symbol)[3] #true or false
    kill_size = open_positions(symbol)[2] #size that is open

    print(f'openposi {openposi}, long {long}, size {kill_size}')

    while openposi == True:

        print('starting kill switch loop til limit fil..')
        temp_df = pd.DataFrame()
        print('just made a temp df')

        phemex.cancel_all_orders(symbol)
        openposi = open_positions(symbol)[1]
        long = open_positions(symbol)[3] #t or false
        kill_size = open_positions(symbol)[2]
        kill_size = int(kill_size)

        ask = ask_bid(symbol)[0]
        bid = ask_bid(symbol)[1]

        if long == False:
            phemex.create_limit_buy_order(symbol, kill_size, bid, params)
            print(f'just made a BUY to CLOSE order of {kill_size} {symbol} at ${bid}')
            print('sleeping for 30 seconds to see if it fills..')
            time.sleep(30)
        elif long == True:
            phemex.create_limit_sell_order(symbol, kill_size, ask, params)
            print(f'just made a SELL to CLOSE order of {kill_size} {symbol} at ${ask}')
            print('sleeping for 30 seconds to see if it fills..')
            time.sleep(30)

        else:
            print('++++ SOMETHING I DIDNT EXPECT IN WILL SWITCH FUNCTION')

        openposi = open_positions(symbol)[1]

def df_sma(symbol=symbol, timeframe = timeframe, limit = limit, sma=sma):
    print('starting indis...')

    bars = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit = limit)
    #print(bars)

    #pandas
    df_sma = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_sma['timestamp'] = pd.to_datetime(df_sma['timestamp'], unit = 'ms')

    #Daily SMA - 20 day
    df_sma[f'sma{sma}_{timeframe}'] = df_sma.close.rolling(sma).mean()

    #if bid < the 20 day sma then = Bearish, if bid > 20 day sma = Bullish
    bid = ask_bid(symbol)[1]

    #if sma > bid = SELL, if sma < bid = BUY
    df_sma.loc[df_sma[f'sma{sma}_{timeframe}'] > bid, 'sig'] = 'SELL'
    df_sma.loc[df_sma[f'sma{sma}_{timeframe}'] < bid, 'sig'] = 'BUY'

    df_sma['support'] = df_sma[:-1]['close'].min()
    df_sma['resis'] = df_sma[:-1]['close'].max()

    print(df_sma)

    return df_sma
