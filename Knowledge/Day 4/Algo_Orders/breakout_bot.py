###### Coding trading bot #2 - breakout bot 2024
'''
calc the last 3 days of data
15m * 3days
96 15m periods in 1d, we want 3 day so 96*3 = 288 + 1 289
find the support and resistance pb 15mins
on retest place orders
'WIFUSDT'

notes
- add support/resis to nice_funcs
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

index_pos = 3 #CHANGE BASED ON WHAT ASSET

#the time between trades
pause_time = 10

# for volume calc Vol_repeat * vol_time == TIME of volume collection
vol_repeat = 11
vol_time = 5

params = {'timeInForce': 'PostOnly',}
vol_decimal = .4

#PULL IN ASK AND BID
askbid = n.ask_bid(symbol)
ask = askbid[0]
bid = askbid[1]
print(f'for {symbol}.. ask: {ask} | bid {bid}')

#PULL IN THE DF_SMA - cause has all data we need
    #call: df_sma(symbol, timeframe, limit,sma) #if not passed uses default
df_sma = n.df_sma(symbol,'15m',289,20) #289 15m

#PULL IN OPEN POSITIONS
#returns: open_positions() open_positions,openpos_bool,openpos_size, long, index_pos
open_pos = n.open_positions(symbol)

#CALCULATE SUPPORT AND RESISTANCE BASED ON CLOSE
curr_support = df_sma['close'].min()
curr_resis = df_sma['close'].max()
print(f'support {curr_support} | resis {curr_resis}')

#RUN BOT

#CALC THE RETEST, WHERE WE PUT ORDERS
#retest() buy_break_out, sell_break_down
def retest():
    print('creating retest number...')
    '''
    if support breaks - SHORT, place asks right below(.1% == .001)
    if resis breaks - LONG, place bid right above(.1% == .001)
    '''

    buy_break_out = False
    sell_break_out = False
    breakoutprice = False
    breakdownprice = False

    #may want to do this on the bid...
    # if most current df resis <= df_wolast:

    if bid > df_sma['resis'].iloc[-1]:
        print(f'we are BREAKING UPWORDS... Buy at previous resis {curr_resis}')
        buy_break_out = True
        breakoutprice = int(df_sma['resis'].iloc[-1] * 1.001)
        ##SWITCH BELOW BACK t >
    elif bid < df_sma['support'].iloc[-1]:
        print(f'we are BREAKING DOWN... Buy at previous support {curr_support}')
        sell_break_out = True
        breakdownprice = int(df_sma['support'].iloc[-1] * .999)

    return buy_break_out, sell_break_out,breakoutprice, breakdownprice


def bot():

    #PULL IN PNL CLOSE
    #returns: pnl_close() [0] pnlclose and [1] in_pos [2] size [3]long TF // just symbol
    pnl_close = n.pnl_close(symbol)

    #FUNCTION SLEEP ON CLOSE
    #returns nothing
    #sleep_on_close(symbol=symbol, pause_time=pause_time) #pause in mins
    sleep_on_close = n.sleep_on_close(symbol,pause_time)

    askbid = n.ask_bid(symbol)
    ask = askbid[0]
    bid = askbid[1]

    re_test = retest()
    break_out = re_test[0]
    break_down = re_test[1]
    breakoutprice = re_test[2]
    breakdownprice = re_test[3]
    print(f'breakout {break_out} {breakoutprice} | breakdown {break_down} {breakdownprice}')

    in_pos = open_pos[1]
    curr_size = open_pos[2]
    curr_size = int(curr_size)
    curr_p = bid

    print(f'for {symbol} breakout {break_out} | breakdown {break_down} | inpos {in_pos} | size {curr_size} | price {curr_p}')

    if (in_pos == False) and (curr_size < pos_size):

        phemex.cancel_all_orders(symbol)
        askbid = n.ask_bid(symbol)
        ask = askbid[0]
        bid = askbid[1]

        if break_out == True:
            print('making an opening order as a BUY')
            print(f'{symbol} buy order of {pos_size} submitted @ {breakoutprice}')
            phemex.create_limit_buy_order(symbol, pos_size, breakoutprice, params)
            print('order submitted so sleeping for 2 mins..')
            time.sleep(120)
        elif break_down == True:
            print('making an opening order as a SELL')
            print(f'{symbol} sell order of {pos_size} submitted @ {breakdownprice}')
            phemex.create_limit_sell_order(symbol,pos_size, breakdownprice,params)
            print('order submitted so sleeping for 2mins...')
            time.sleep(120)
        else:
            print('not submitting any orders cuz no break out or down.. sleeping 1min')
            time.sleep(60)

    else:
        print('we are in a position already so not making any orders')

schedule.every(28).seconds.do(bot)

while True:
    try:
        schedule.run_pending()
    except:
        print('+++++++++++++MAYBE AN INTERNET PROBLEM, CODE FAILED.. sleep 30..')
        time.sleep(30)