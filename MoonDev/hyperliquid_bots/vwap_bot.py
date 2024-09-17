'''VWAP BOT'''
import dontshare as d
import nice_funcs as n
from eth_account.signers.local import LocalAccount
import eth_account
import json,time,random
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import ccxt
import pandas as pd
import datetime
import schedule
import requests

symbol = 'WIF'
timeframe = '15m'
sma_window = 20
lookback_days = 1
size = 1
target = 8
max_loss = -10
leverage = 2
max_positions = 1


def bot():
    secret_key = d.private_key

    account1 = LocalAccount = eth_account.Account.from_key(secret_key)
    
    positions1, im_in_pos, mypos_size, pos_sym1, entry_px1, pnl_perc1, long1, num_of_pos = n.get_position_andmaxpos(symbol,account1,max_positions)
    print(f'these are my positions for {symbol} {positions1}')

    lev,pos_size = n.adjust_leverage_size_signal(symbol, leverage, account1)

    if im_in_pos:
        n.cancell_all_orders(account1)
        print('in position so check pnl close')
        n.pnl_close(symbol, target,max_loss, account1)

    else:
        print('not in position so no pnl close')

    ask,bid, l2_data = n.ask_bid(symbol)

    # 11th bid and ask
    bid11 = float(l2_data[0][10]['px'])
    ask11 = float(l2_data[1][10]['px'])
    
    #get vwap
    latest_vwap = n.calculate_vwap_with_symbol(symbol)[1]

    print(f'the latest vwap is {latest_vwap}')

    random_chance = random.random()

    if bid > latest_vwap:
        if random_chance <= 0.7: #70% of chance
            goin_long = True
            print(f'price is above vwap {ask} > {latest_vwap}, going long')
        else:
            goin_long = False
            print(f'price is above vwap {ask} < {latest_vwap}, but not going long')

    else:
        if random_chance <= 0.3: #30% chance
            going_long = True
            print(f'price is below vwap {ask} < {latest_vwap}, going long')
        else:
            going_long = False
            print(f'price is below vwap {ask} < {latest_vwap}, not going long')

    #ENTER ORDER IF NO POSITION
    if not im_in_pos and goin_long:
        print(f'not in position so we are quoteing at buying @{bid11}')
        n.cancell_all_orders(account1)
        print('just cancelled all orders')

        #enter buy order
        n.limit_order(symbol,True, pos_size, bid11, False, account1)
        print(f'just placed an order for {pos_size} at {bid11}')

    elif not im_in_pos and not goin_long:

        print(f'not in position.. we are quoting at sell @{ask11}')
        n.cancell_all_orders(account1)
        print('just cancelled all orders')

        #enter sell order
        n.limit_order(symbol,False,pos_size, ask11,False,account1)
        print(f'just placed an order for {pos_size} at @{ask11}')

    else:
        print(f'our position is {im_in_pos}')

bot()
schedule.every(28).seconds.do(bot)

while True:
    try:
        schedule.run_pending()
        time.sleep(10)
    except Exception as e:
        print('+++++++MAYBE INTERNET CONNECTION OR SOMETHING')
        print(e)
        time.sleep(30)