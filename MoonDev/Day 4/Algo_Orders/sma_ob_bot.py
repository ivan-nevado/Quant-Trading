###### Coding trading bot #1 - sma bot w/ob data 2024

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

#ask_bid()[0] = ask, [1] = bid
def ask_bid():

    ob = phemex.fetch_order_book(symbol)
    #print(ob)

    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]

    return ask, bid #ask_bid()[0] = ask, [1] = bid

#daily_sma()[0] = df_d # which is the daily sma
def daily_sma():

    print('starting indis...')

    timeframe = '1d'
    num_bars = 100

    bars = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=num_bars)
    #print(bars)
    df_d = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_d['timestamp'] = pd.to_datetime(df_d['timestamp'], unit = 'ms')

    #Daily SMA -20day
    df_d['sma20_d'] = df_d.close.rolling(20).mean()

    #if bid  the 20 day sma then = bearish, if bid > 20 day sma = bullish
    bid = ask_bid()[1]

    #if sma > bid = Sell, if sma < bid = Buy
    df_d.loc[df_d['sma20_d'] > bid, 'sig'] = 'SELL'
    df_d.loc[df_d['sma20_d'] < bid, 'sig'] = 'BUY'

    #print(df_d)

    return df_d

#daily_sma()[0] = df_f which is the 15m sma
def f15_sma():

    print('starting the 15 min sma...')

    timeframe = '15m'
    num_bars = 100

    bars = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=num_bars)
    #print(bars)
    df_f = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low','close', 'volume'])
    df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], unit = 'ms')

    #Daily SMA -20 day
    df_f['sma20_15'] = df_f.close.rolling(20).mean()

    #Buy price 1+2 and Sell price 1+2 (then later figure out which I chose)
    #Buy/Sell to open around the 15m sma (20day) - .1% under and .3% over
    df_f['bp_1'] = df_f['sma20_15'] * 1.001 # 15m sma .1% under and .3% over
    df_f['bp_2'] = df_f['sma20_15'] * .997
    df_f['bp_3'] = df_f['sma20_15'] * .999
    df_f['bp_4'] = df_f['sma20_15'] * 1.003

    # print(df_f)

    return df_f

#open_positions() open_positions, openpos_bool, openpos_size, long
def open_positions():
    params = {'type':'swap','code':'USDT'}
    phe_bal = phemex.fetch_balance(params=params)
    open_positions = phe_bal['info']['data']['positions']
    #print(open_positions)
    openpos_side = open_positions[0]['side']
    openpos_size = open_positions[0]['size']

    if openpos_side == ('Buy'):
        openpos_bool = True
        long = True
    elif openpos_side == ('Sell'):
        openpos_bool = True
        long = False
    else:
        openpos_bool = False
        long = None

    return open_positions, openpos_bool, openpos_size, long

def kill_switch():

    #gracefully limit close us

    #open_positions() open_positions, openpos_bool, openpos_size,long

    print('starting the kill switch')
    openposi = open_positions()[1] #true or false
    long = open_positions()[3] #true or false
    kill_size = open_positions()[2]

    print(f'openposi {openposi}, long {long}, size {kill_size}')

    while openposi == True:
        
        print('starting the kill switch loop till limit fil...')
        temp_df = pd.DataFrame()
        print('just made a temp df')

        phemex.cancel_all_orders(symbol)
        openposi = open_positions()[1]
        long = open_positions()[3]
        long = open_positions()[3] #true or false
        kill_size = open_positions()[2]
        kill_size = int(kill_size)

        ask = ask_bid()[0]
        bid = ask_bid()[1]

        if long == False:
            phemex.create_limit_buy_order(symbol, kill_size, bid, params)
            print(f'just made a BUY to CLOSE order of {kill_size} {symbol} at ${bid}')
            print('sleeping for 30 seconds to see if it fills...')
            time.sleep(30)
        elif long == True:
            phemex.create_limit_sell_order(symbol,kill_size,ask,params)
            print(f'just made a SELL to CLOSE order of {kill_size} {symbol} at ${ask}')
            print('sleeping for 30 seconds to see it fills...')
            time.sleep(30)
        else:
            print('+++++++SOMETHING I DIDNT EXPECT IN KILL SWITCH FUNCTION')

        openposi = open_positions()[1]

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

def ob():

    print('fetching order book data...')

    df = pd.DataFrame()
    temp_df = pd.DataFrame()

    ob = phemex.fetch_order_book(symbol)
    #print(ob)
    bids = ob['bids']
    asks = ob['asks']

    first_bid = bids[0]
    first_ask = asks[0]

    bid_vol_list = []
    ask_vol_list = []

    #if SELL vol > BUY vol AND profit target hit, exit

    # get last 1min of volume.. and if sell > buy vol do x

    for x in range(11):

        for set in bids:
            #print(set)
            price = set[0]
            vol = set[1]
            bid_vol_list.append(vol)
            #print(price)
            #print(vol)

            #print(bid_vol_list)
            sum_bidvol = sum(bid_vol_list)
            #print(sum_bidvol)
            temp_df['bid_vol'] = [sum_bidvol]

        for set in asks:
            #print(set)
            price = set[0] # [40000, 344]
            vol = set[1]
            ask_vol_list.append(vol)
            #print(price)
            #print(vol)

            sum_askvol = sum(ask_vol_list)
            temp_df['ask_vol'] = [sum_askvol]

        #print(temp_df)
        time.sleep(5) #change backt to 5 later
        df = df.append(temp_df)
        print(df)
        print(' ')
        print('-----')
        print(' ')
    print('done calculating volumen for bids and asks...')
    print('calculating the sums')
    total_bidvol = df['bid_vol'].sum()
    total_askvol = df['ask_vol'].sum()
    print(f'last 1m this is total Bid Vol: {total_bidvol} | {total_askvol}')

    if total_bidvol > total_askvol:
        control_dec = (total_askvol/total_bidvol)
        print(f'Bulls are in control: {control_dec}...')
        #if bulls are in control, use regular target
        bullish = True
    else:

        control_dec = (total_bidvol / total_askvol)
        print(f'Bears are in control: {control_dec}...')
        bullish = False
        #.2, .36, .16

    open_posi = open_positions()
    openpos_tf = open_posi[1]
    long = open_posi[3]
    print(f'openpos_tf: {openpos_tf} || long: {long}')

    if openpos_tf == True:
        if long == True:
            print('we are in a long position...')
            if control_dec < vol_decimal: #vol_decimal set to .4 at top
                vol_under_dec = True
            else:
                print('volumen is not under dec so setting vol_under_dec to False')
                vol_under_dec = False

        else:
            print('we are in a short position')
            if control_dec < vol_decimal: #vol_decimal set to .4 at top
                vol_under_dec = True
                #print('going to sleep for a minute.. cuz under vol decimal')
                #time.sleep(6) # change to 60
            else:
                print('volume is not under dec so setting vol_under_dec to False')
                vol_under_dec = False

    else:
        print('we are not in a position')

    #when vol_under_dec == FALSE AND target hit, then exit
    print(vol_under_dec)

    return vol_under_dec

# pnl close() [0] pnl close and [1] in_pos [2] size [3] long TF
def pnl_close(symbol=symbol, target=target,max_loss = max_loss):
    
    print(f'checking to see if its time to exit...')

    params = {'type':"swap", 'code':'USDT'}
    pos_dict = phemex.fetch_positions(params=params)
    #print(pos_dict)
    pos_dict = pos_dict[0] 
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
    print(f'this is our PNL percentage: {(perc)}%')

    pnlclose = False
    in_pos = False

    if perc > 0:
        in_pos = True
        print(f'we are in a winning position')
        if perc > target:
            print(':) :) we are in profit & hit target.. checking volume to see if we should start kill switch')
            pnl_close = True
            vol_under_dec = ob()
            if vol_under_dec == True:
                print(f'volume is under the decimal threshold we set of {vol_decimal}... so sleeping 30s')
                time.sleep(30)
            else:
                print(f':) :) starting the kill switch because we hit our target of {target}% and already checked vol...')
                kill_switch()
            
        else:
            print('we have not hit our target yet')

    elif perc < 0: # -10, -20,

        in_pos = True

        if perc <= max_loss: #under -55, -5
            print(f'we need to exit now down {perc}... so starting the kill switch.. max loss {max_loss}')
            kill_switch()
        else:
            print(f'we are in a losing position of {perc}.. but chilling cause max loss is {max_loss}')

    else:
        print('we are not in a position')

    if in_pos == True:

        #if breaks over .8% over 15m sma, then close pos(STOP LOSS)

        #pull in 15m sma
        df_f = f15_sma()
        #print(df_f)
        #df_f['sma20_15'] #last value of this
        last_sma15 = df_f.iloc[-1]['sma20_15']
        last_sma15 = int(last_sma15)
        print(last_sma15)
        #pull current bid
        curr_bid = ask_bid()[1]
        curr_bid = int(curr_bid)
        print(curr_bid)

        sl_val = last_sma15 * 1.008
        print(sl_val)

        #TURN KILL SWITCH ON

        #5/11 - removed the below and implementing a 55% stop loss
            #in the pnl section
        # if curr_bid > sl_val:
        #     print('current bid is above stop loss value.. starting kill switch..')
        #     kill_switch()
        # else:
        #     print('chillen in position..')
                
    else:
        print('we are not in a position...')      

    print(f'just finished checking the PNL close..')

    return pnlclose, in_pos, size, long

def bot():

    pnl_close() #checking if we hit our pnl
    sleep_on_close() #checking sleep on close

    df_d = daily_sma() #determines LONG/SHORT
    df_f = f15_sma() #provides prices bp_1,bp_2,sp_1,sp_2
    ask = ask_bid()[0]
    bid = ask_bid()[1]

    #MAKE OPEN ORDER
    #LONG OR SHORT?
    sig = df_d.iloc[-1]['sig']
    #print(sig)

    open_size = pos_size/2

    #ONLY RUN IF NOT IN POSITION
    #pnl close() [0] pnl close and [1] in_pos
    in_pos = pnl_close()[1]
    print(in_pos)
    curr_size = open_positions()[2]
    curr_size = int(curr_size)
    print(curr_size)

    curr_p = bid
    last_sma15 = df_f.iloc[-1]['sma20_15']
    #pos_size = 50, if in_pos == False

    #never get in a position bigger than pos_size(52)

    if(in_pos == False) and (curr_size < pos_size):

        phemex.cancel_all_orders(symbol)

        #fix order function so I stop sending order if price > sma
        if(sig == 'BUY') and (curr_p > last_sma15):

            #buy sma daily is  price == BUY
            print('making an opening order as a BUY')
            bp_1 = df_f.iloc[-1]['bp_1']
            bp_2 = df_f.iloc[-1]['bp_2']
            print(f'this is bp_1: {bp_1} this is bp_2: {bp_2}')
            phemex.cancel_all_orders(symbol)
            phemex.create_limit_buy_order(symbol,open_size,bp_1,params)
            phemex.create_limit_buy_order(symbol,open_size,bp_2,params)

            print('just made opening order so going to sleep for 2mins..')
            time.sleep(120)
        elif (sig == 'SELL') and (curr_p < last_sma15):
            print('making an open order as a SELL')
            sp_1 = df_f.iloc[-1]['sp_1']
            sp_2 = df_f.iloc[-1]['sp_2']
            print(f'this is sp_1: {sp_1} this is sp_2: {sp_2}')
            phemex.cancel_all_orders(symbol)
            phemex.create_limit_sell_order(symbol,open_size,sp_1,params)
            phemex.create_limit_sell_order(symbol,open_size,sp_2,params)
            print('just made opening order so going to sleep for 2mins..')
        else:
            print('not submitting orders... price prob higher or lower than sma.. 10m sleep...')
            time.sleep(600)

    else:
        print('we are in a position already so not making new orders')

schedule.every(28).seconds.do(bot)

while True:
    try:
        schedule.run_pending()
    except:
        print('+++++MAYBE AN INTERNET PROB OR SOMETHING')
        time.sleep(30)