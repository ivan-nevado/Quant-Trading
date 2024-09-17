import pandas as pd
import datetime
import os
import ccxt
import dontshare as d
from math import ceil

symbol = 'BTC/USD'
timeframe = '1h'
weeks = 100

# Function to convert timeframe to seconds
def timeframe_to_sec(timeframe):
    if 'm' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60
    elif 'h' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60 * 60
    elif 'd' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 24 * 60 * 60
    

def get_historical_data(symbol, timeframe, weeks):

    if os.path.exists(f'{symbol}{timeframe}{weeks}.csv'):
        return pd.read_csv(f'{symbol}{timeframe}{weeks}.csv')

    now = datetime.datetime.utcnow()
    coinbase = ccxt.coinbase({
        'apiKey': d.api_key,
        'secret': d.api_secret,
        'enableRateLimit': True,
    })

    granularity = timeframe_to_sec(timeframe) # Convert timeframe to seconds

    total_time = weeks * 7 * 24 * 60 * 60
    run_times = ceil(total_time / (granularity * 200))

    dataframe = pd.DataFrame()

    for i in range(run_times):

        since = now - datetime.timedelta(seconds=granularity * 200 * (i + 1))
        since_timestamp = int(since.timestamp()) * 1000  # Convert to milliseconds

        data = coinbase.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=200)
        df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        dataframe = pd.concat([df, dataframe])

    dataframe = dataframe.set_index('datetime')
    dataframe = dataframe[["open", "high", "low", "close", "volume"]]
    dataframe.to_csv(f'{symbol[0:3]}-{timeframe}-{weeks}wks-data.csv')

    return dataframe

# dataframe.to_csv(f'storage/{symbol}{timeframe}{weeks}.csv')
print(get_historical_data(symbol, timeframe, weeks))