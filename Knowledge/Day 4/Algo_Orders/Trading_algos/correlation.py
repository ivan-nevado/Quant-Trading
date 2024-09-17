'''After ETH does a move the bot looks at other altcoins to replicate the move
And uses the one with more delay
'''
import ccxt,config,schedule,cbpro
from functions import *

coinbase = cbpro.PublicClient()

phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': config.api_key,
    'secret': config.api_secret
})

#config settings for bot
timeframe = 900 #timeframe of candles in 5 seconds(5min would be 300, 15min would be 900,etc)
dataRange = 20 #amount of candles to get
sl_percent = .2 #percent to take loss on trades
tp_percent = .25 #percent to take profit on trades
size = 1 #set amount to buy
params = {'timeInForce':'PostOnly', 'takeProfit':0,'stopLoss':0} #set default tp and sl, will be changed when an order is about to be placed

#this is the symbol we are checking on coinbase
symbol = 'ETH-USD' #coinbase differs from Phemex
symbol_phe = 'ETHUSD' #phemex symbol

#these are the coins that will actually trade on
alt_coins = ['ADAUSD','DOTUSD','MANAUSD','XRPUSD','UNIUSD','SOLUSD']

def bot():
    #get price and candles for eth
    price = float(coinbase.get_product_ticker(symbol)['bid']) #get the current bid
    candles = get_candle_df(coinbase,symbol,timeframe,dataRange) #get candles dataFrame

    #calculate our signals
    trange = calc_tr(candles)
    support,resistance = calc_sup_res(candles,dataRange)
    print('line 38')

    #if price goes above the true range or resistance
    #if it has broken TR or resistance then loop through the above alts
    if price > candles.close.iloc[-1] + trange or price > resistance:
        #get the current prices for the alt coins
        coinData = {}
        #we add all data to a df after looping through all the alts %gained
        for coin in alt_coins:
            cur_price = float(phemex.fetch_ticker(coin)['bid']) # gets coin current price
            coinData[coin] = (abs(cur_price - candles.close.ilo[-1]) / candles.close.iloc[-1]) * 100.0 # get perc change from last candle

        #finding the alt with least movement, meaning its lagging the most

        params['stopLoss'] = price * (1-(sl_percent/100)) #set stop loss price
        params['takeProfit'] = price * (1+(tp_percent/100)) #set take profit price
        order = phemex.create_limit_buy_order(symbol_phe,size,price=price,params=params) #place order

    #if price goes below the true range or support
    #other wise we do the opposite, and do this going short
    elif price < candles.close.iloc[-1]-trange or price < support:
        #get the current prices for altcoins
        coinData = {}
        for coin in alt_coins:
            cur_price = float(phemex.fetch_ticker(coin)['bid']) #get current price
            coinData[coin] = (abs(cur_price - candles.close.iloc[-1]) / candles.close.iloc[-1]) * 100.0 # get the perc change from last candle

        most_lagging = min(coinData, key = coinData.get) #get the coin with the min change

        params['stopLoss'] = price * (1+(sl_percent/100)) #set stop loss price
        params['takeProfit'] = price * (1-(tp_percent/100)) #set take profit price
        order = phemex.create_limit_sell_order(symbol_phe,size,price=price,params=params)
    print('line 70')

bot()

schedule.every(20).seconds.do(bot)

while True:
    try:
        schedule.run_pending()
    except:
        print('++++++ ERROR RUNNING BOT, SLEEPING FOR 30SECS')
        time.sleep(30)
