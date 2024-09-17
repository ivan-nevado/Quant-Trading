'''
Bollinger Band Bot
'''

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

secret_key = d.private_key

def bot():

    account1 = LocalAccount = eth_account.Account.from_key(secret_key)

    positions1, im_in_pos, mypos_size, pos_sym1, entry_px1, pnl_perc1, long1, num_of_pos = n.get_position_andmaxpos(symbol,account1,max_positions)

    print(f'these are positions for {symbol} {positions1}')

    lev, pos_size = n.adjust_leverage_size_signal(symbol,leverage,account1)

    #dividing position by 2
    pos_size = pos_size / 2

    if im_in_pos:
        n.cancell_all_orders(account1)
        print('in position so check pnl close')
        n.pnl_close(symbol,target, max_loss,account1)
    else:
        print('not in position so no pnl close')

    #get price
    ask, bid, l2_data = n.ask_bid(symbol)

    print(f'ask: {ask} bid: {bid}')

    bid11 = float(l2_data[0][10]['px'])
    ask11 = float(l2_data[1][10]['px'])

    print(f'ask11: {ask11} bid11: {bid11}')

    snapshot_data = n.get_ohlcv2('BTC', '1m', 500)
    df = n.process_data_to_df(snapshot_data)
    bbdf = n.calculate_bollinger_bands(df)
    bollinger_bands_tight = n.calculate_bollinger_bands(df)

    print(f'bollinger bands are tight: {bollinger_bands_tight}')

    #ONLY ENTERS IF BOLLINGER BANDS ARE TIGHT
    if not im_in_pos and bollinger_bands_tight:
        print('bollinger bands are tight and we dont have a position so entering')
        print(f'not in position we are quoting a sell @{ask} and buy @{bid}')
        #cancell all orders
        n.cancell_all_orders(account1)

        print('just canceled all orders')

        #ENTER BUY ORDER
        n.limit_order(symbol,True,pos_size,bid11, False, account1)
        print(f'just placed an order for {pos_size} at {bid}')

        #ENTER SELL ORDER
        n.limit_order(symbol,False,pos_size,ask11, False, account1)
        print(f'just placed an order for {pos_size} at {ask}')

    elif bollinger_bands_tight == False:
        n.cancell_all_orders(account1)
        n.close_all_positions(account1)
    else:
        print(f'our position is {im_in_pos} bollinger bands may not be tight')
        

bot()
schedule.every(30).seconds.do(bot)

while True:
    try:
        schedule.run_pending()
        time.sleep(10)
    except Exception as e:
        print('*** maybe internet connection lost... sleeping 30 secs and retrying')
        print(e)
        time.sleep(30)