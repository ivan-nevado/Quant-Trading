'''
TA -LIB a package that gives a bunch of technical analysis indicators
talib docs - https://ta-lib.github/ta-lib-python/doc_index.html
you can une this indicators for backtesting and for bots

pandas_TA review
documentation- https://github.com/twopirllc/pandas-ta
'''
import talib as ta
import pandas_ta as pta
import pandas as pd

#get data
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\hyperliquid_bots\MAN-1h-100wks-data.csv')

print(df.head())

#ta-lib functions#

#SMA
# df['sma'] = ta.SMA(df['close'], timeperiod = 20)


#RSI
# df['rsi'] = ta.RSI(df['close'], timeperiod = 14)

#EMA
# df['ema'] = ta.EMA(df['close'], timeperiod = 10)


#bollinger_bands
# df['boll_upper'], df['boll_mid'],df['boll_lower'] = ta.BBANDS(df['close'], timeperiod = 20, nbdevup = 2, nbdevdn=2, matype = 0)

#MACD
# df['macd_line'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'], fastperiod = 12, slowperiod = 26, signalperiod = 9)

#ATR -average true range
# df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod = 14)

#stochastic osciallator
# df['stock_k'],df['stock_d'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period = 14, slowk_period = 3, slowk_matype = 0, slowd_period = 3, slowd_matype = 0)

#commodity channel osciallator
# df['cci_20'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod = 20)

#parabolic sar
# df['sar'] = ta.SAR(df['high'], df['low'], acceleration = 0.02, maximum = 0.2)

#obs - on balace volume
# df['obv'] = ta.OBV(df['close'], df['volume'])

#pandas-ta functions#
#df['sma_10] = pta.sma(df['close'], length = 10)
#df['ema_10] = pta.sma(df['close], length = 10)
# df['rsi_14'] = ta.rsi(df['close'], length = 14)
# df[['macd_line', 'macd_signal', 'macd_hist']] = pta.macd(df['close'], fast = 12, slow = 26, signal = 9)
df[['stoch_k', 'stoch_d']] = pta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
print (df)
help(df.ta)






