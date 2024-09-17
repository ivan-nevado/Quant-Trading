############## Coding Algo Orders

# connect exchange

import ccxt 
import key_file as k
import time, schedule



phemex = ccxt.phemex ({
    'enableRateLimit':True,
    'apiKey': k.key,
    'secret': k.secret
})



bal = phemex.fetch_balance()
# print(bal)

symbol = 'BTCUSDT'
size = 0.000288
bid = 57500
# params = {'timeInForce': 'PostOnly',}

# Fetch account balance and open positions
params = {'type': 'swap', 'code': 'USDT'}
balance_info = phemex.fetch_balance(params=params)

# Inspect the data structure
positions = balance_info['info']['data']['positions']

# Print out the symbols with their corresponding index
for index, position in enumerate(positions):
    symbol = position['symbol']
    print(f"Index: {index}, Symbol: {symbol}")

# # making an order

# order = phemex.create_limit_buy_order(symbol, size, bid, params)
# print(order)

# # # cancel orders
# phemex.cancel_all_orders(symbol)

# # order for a couple of seconds and then cancel it

# def bot():
#     phemex.create_limit_buy_order(symbol,size,bid,params)
#     time.sleep(10)
#     phemex.cancel_all_orders(symbol)


# schedule.every(2).seconds.do(bot)

# while  True:
#      try:
#          schedule.run_pending()
#      except:
#          print('+++++ MAYBE AN INTERNET PROB OR SOMETHING')
#          time.sleep(30)
    