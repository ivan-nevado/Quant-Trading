import asyncio
import json
import os
from datetime import datetime
import pytz
from websockets import connect
from termcolor import cprint

# List of symbols you want to track
symbols = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt', 'dogeusdt', 'wifusdt']
websocket_base_url = 'wss://stream.binance.com:9443/ws'

# Define TradeAggregator class
class TradeAggregator:
    def __init__(self):
        self.trade_buckets = {}

    async def add_trade(self, symbol, timestamp, usd_size, is_buyer_maker):
        trade_key = (symbol, timestamp, is_buyer_maker)
        self.trade_buckets[trade_key] = self.trade_buckets.get(trade_key, 0) + usd_size

    async def check_and_print_trades(self):
        timestamp_now = datetime.now(pytz.timezone('Europe/Madrid')).strftime("%H:%M:%S")
        deletions = []
        for trade_key, usd_size in self.trade_buckets.items():
            symbol, timestamp, is_buyer_maker = trade_key
            if timestamp < timestamp_now and usd_size > 500000:
                trade_type = "BUY" if not is_buyer_maker else "SELL"
                back_color = 'on_blue' if not is_buyer_maker else 'on_magenta'

                display_size = usd_size / 1000000
                attrs = ['bold']
                if usd_size > 3000000:
                    attrs.append('blink')
                    cprint(f"{trade_type} {symbol} {timestamp} ${display_size:.2f}m", 'white', back_color, attrs=attrs)
                else:
                    cprint(f"{trade_type} {symbol} {timestamp} ${display_size:.2f}m", 'white', back_color, attrs=attrs)

                deletions.append(trade_key)

        for key in deletions:
            del self.trade_buckets[key]

trade_aggregator = TradeAggregator()

# Define async function to handle trades
async def trade_handler(symbol, trade_aggregator):
    uri = f"{websocket_base_url}/{symbol.lower()}@aggTrade"
    async with connect(uri) as websocket:
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                usd_size = float(data['p']) * float(data['q'])
                trade_time = datetime.fromtimestamp(data['T'] / 1000, pytz.timezone('Europe/Madrid')).strftime('%H:%M:%S')

                await trade_aggregator.add_trade(symbol.upper().replace('USDT', ''), trade_time, usd_size, data['m'])

            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(1)

# Define function to print aggregated trades every second
async def print_aggregated_trades_every_second(aggregator):
    while True:
        await asyncio.sleep(1)
        await aggregator.check_and_print_trades()

# Define main function
async def main():
    tasks = [trade_handler(symbol, trade_aggregator) for symbol in symbols]
    tasks.append(print_aggregated_trades_every_second(trade_aggregator))
    await asyncio.gather(*tasks)

# Run the main function
asyncio.run(main())


                    
