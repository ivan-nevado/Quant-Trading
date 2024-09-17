'''
Engulfing close== if last candle is lower than prev candle low
trade to whatever side is engulfed
'''

import ccxt
import json
import pandas as pd
import numpy as np
import key_file as k
from datetime import date,datetime, timezone, tzinfo
import time, schedule
import nice_funcs as n

phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': k.key,
    'secret': k.secret
})

symbol = 'WIFUSDT'
pos_size = 30 #125, 75,
target = 9 #%gain I want
max_loss = -8

params = {'timeInForce': 'PostOnly',}

def bot():
    n.pnl_close(symbol, target, max_loss)

    timeframe = '15m'
    limit = 97
    sma = 20
    df = n.df_sma(symbol, timeframe, limit, sma)
    print(df)

    askbid = n.ask_bid(symbol)
    ask = askbid[0]
    bid = askbid[1]

    #position info

    allposinfo = n.open_positions(symbol)
    in_pos = allposinfo[1]
    curr_size = allposinfo[2]
    curr_size = int(curr_size)

    sma20_15m = df.iloc[-1][f'sma{sma}_{timeframe}']
    last_close = df.iloc[-1]['close']
    print(f'inpos {in_pos}, curr_size {curr_size}, bid {bid}, ask {ask}, sma20_15 {sma20_15m}, last close {last_close}')

    #if bid > close[-1] - take a long and (if bid > sma20_15) aka if sig == BUY
    if(bid > last_close) and (bid > sma20_15m):
        print('make a buy to open order...')
        phemex.cancel_all_orders(symbol)
        askbid = n.ask_bid(symbol)
        ask = askbid[0]
        bid = askbid[1]
        orderprice = bid * .999
        phemex.create_limit_buy_order(symbol, pos_size, orderprice, params)
        print('just made a buy order because there was engulfing candle.. sleep 2mins')
        time.sleep(120)

    elif (bid < last_close) and (bid < sma20_15m):
        print('make a sell to open order...')
        phemex.cancel_all_orders(symbol)
        askbid = n.ask_bid(symbol)
        ask = askbid[0]
        bid = askbid[1]
        orderprice = ask * 1.001
        phemex.create_limit_sell_order(symbol, pos_size, orderprice, params)
        print('just made a sell order because there was engulfing candle.. sleep 2 mins')
        time.sleep(120)

    else:
        print('not making an order')

schedule.every(28).seconds.do(bot)

while True:
    try:
        schedule.run_pending()
    except:
        print('+++++++++++++MAYBE AN INTERNET PROBLEM, CODE FAILED.. sleep 30..')
        time.sleep(30)