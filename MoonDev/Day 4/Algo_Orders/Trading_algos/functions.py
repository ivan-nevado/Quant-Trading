from turtle import position
import pandas as pd
from datetime import datetime, time
from pytz import timezone
import pandas_ta as ta
from time import sleep
import ccxt

def in_timeframe():
    '''
    returns True if the current time is between 9:30-4pm
    '''

    #get current eastern time
    now = datetime.now(timezone('US/Eastern')).time()
    #get day of week as a number (0=monday,6=sunday)
    day = datetime.today().weekday()

    #return if the current time in the ES is bet 9:30-4pm
    if(now >= time(9,30) and now <time(16)) and (day < 5 and day >=0):
        return True
    return False

def get_position(phemex,symbol):
    '''
    get the position for the given symbol
    '''

    params = {'type':'swap','code':'USD'}
    phe_bal = phemex.fetch_balance(params=params)
    