''' this file shows how to build a kill switch and pnl close
for hyper liquid'''

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
max_loss = -5
target = 9
acct_min = 9
timeframe = '4h'
size = 10
coin = symbol
secret_key = d.private_key
account = LocalAccount = eth_account.Account.from_key(secret_key)

acct_min = 12

def bot():

    print('this is our bot')

    print('controlling risk with our pnl close')

    #check pnl close
    n.pnl_close(symbol, target, max_loss, account)

    #if my account size goes under $100, and never $70
    acct_val = float(n.acct_bal(account))

    if acct_val < acct_min:
        print(f'account value is {acct_val} and closing because out low is {acct_min}')
        n.kill_switch(symbol,account)


bot()