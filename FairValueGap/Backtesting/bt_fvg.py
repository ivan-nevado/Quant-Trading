'''This bot will buy when price touches the top of a bullish FVG and sell if it touches the bottom of the bearish FVG
First it will look in at the EMA of 10 periods to see if we are bearish or bullish
If we are bullish it will take bullish FVGs below price but they have to be also above the EMA
If we are bearish it will take bearish FVGs above price but they have to be also below the EMA
'''
import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
import warnings


# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\BTC-6h-100wks-data.csv')

# Ensure 'Date' column exists and is properly formatted
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
# df = df.iloc[320:400]

# Define Fair Value Gap detection function with MitigatedIndex
def detect_fvg(ohlc: pd.DataFrame):
    fvg_signals = np.where(
        (
            (ohlc["High"].shift(1) < ohlc["Low"].shift(-1))
            & (ohlc["Close"] > ohlc["Open"])
        )
        | (
            (ohlc["Low"].shift(1) > ohlc["High"].shift(-1))
            & (ohlc["Close"] < ohlc["Open"])
        ),
        np.where(ohlc["Close"] > ohlc["Open"], 1, -1),
        np.nan,
    )
    
    top = np.where(
        ~np.isnan(fvg_signals),
        np.where(
            ohlc["Close"] > ohlc["Open"],
            ohlc["Low"].shift(-1),
            ohlc["Low"].shift(1),
        ),
        np.nan,
    )
    
    bottom = np.where(
        ~np.isnan(fvg_signals),
        np.where(
            ohlc["Close"] > ohlc["Open"],
            ohlc["High"].shift(1),
            ohlc["High"].shift(-1),
        ),
        np.nan,
    )
    
    # Calculate MitigatedIndex
    mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
    for i in np.where(~np.isnan(fvg_signals))[0]:
        mask = np.zeros(len(ohlc), dtype=np.bool_)
        if fvg_signals[i] == 1:
            mask = ohlc["Low"][i + 2 :] <= top[i]
        elif fvg_signals[i] == -1:
            mask = ohlc["High"][i + 2 :] >= bottom[i]
        if np.any(mask):
            j = np.argmax(mask) + i + 2
            mitigated_index[i] = j

    mitigated_index = np.where(np.isnan(fvg_signals), np.nan, mitigated_index)
    # Create an index array where the FVGs are detected
    fvg_indices = np.full(len(ohlc), np.nan)
    valid_indices = np.where(~np.isnan(fvg_signals))[0]
    fvg_indices[valid_indices] = valid_indices
    return pd.concat(
        [
            pd.Series(fvg_signals, name="FVG"),
            pd.Series(top, name="Top"),
            pd.Series(bottom, name="Bottom"),
            pd.Series(mitigated_index, name="MitigatedIndex"),
            pd.Series(fvg_indices, name="FVGIndex"),
        ],
        axis=1,
    )

def detect_ema(ohlc: pd.DataFrame):
    ema = ohlc['Close'].ewm(span=100, adjust=False).mean()
    return ema

# Function to determine if an FVG is filled
def is_fvg_filled(fvg_row, df, start_index):
    top = fvg_row.Top
    bottom = fvg_row.Bottom
    if pd.isna(start_index) or not isinstance(start_index, int):
         return None
    for i in range(start_index, len(df)):
        if (fvg_row.FVG == 1 and df['Low'].iloc[i] < bottom) or \
           (fvg_row.FVG == -1 and df['High'].iloc[i] > top):
            return i  # Return the index where it is filled
    return None  # Return None if not filledP

class FVGStrategy(Strategy):
    min_fvg_height = 10
    ema_period = 100
    risk_reward_ratio = 3
    sl_offset = 0.001
    def init(self): 
        self.fvgs = detect_fvg(self.data.df)
        self.processed_fvgs = set()
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, adjust=False).mean(), self.data.Close)
        
    def next(self):###Lo que hace es que la misma vela comprueba todos los fvgs o algo así incluso del futuro viendo si cumple la condición
        current_index = len(self.data) - 1  # Get the current candle index
        # Filter FVGs up to the current index
        valid_fvgs = self.fvgs[self.fvgs.FVGIndex <= current_index]
        macro_trend_bullish = self.data.Low[-1] > self.ema[-1]
        for fvg in valid_fvgs.itertuples():
            fvg_type, fvg_top, fvg_bottom, fvg_mit_index, fvg_index = fvg.FVG, fvg.Top, fvg.Bottom, fvg.MitigatedIndex, fvg.FVGIndex
            
             # Check if the FVG height meets the minimum requirement
            fvg_height = abs(fvg_top - fvg_bottom)
            if fvg_height < self.min_fvg_height:
                continue

            # Ensure indices are valid before proceeding
            if pd.isna(fvg_index) or fvg_index in self.processed_fvgs:
                continue

            filled_index = is_fvg_filled(fvg, df, fvg_index + 2)
                
            if filled_index is not None:
                continue
            
            fvg_mit_index = int(fvg_mit_index)  # Convert to integer index
            if macro_trend_bullish:
                if fvg_type == 1:
                    # Check if the current candle is the mitigation index
                    if current_index == fvg_mit_index:
                        # Check if the current price has touched or crossed the FVG top
                        if self.data.High[-1] >= fvg_top and fvg_bottom > self.ema[-1]:
                            entry_price = fvg_top  # Use the FVG top as the entry price
                            stop_loss = fvg_bottom * (1 - self.sl_offset)
                            take_profit = entry_price + ((entry_price - stop_loss) * self.risk_reward_ratio)
                        
                            if stop_loss < entry_price < take_profit:
                                self.buy(sl=stop_loss, tp=take_profit, size=0.1, limit=entry_price)
                                
                                self.processed_fvgs.add(fvg_index)  # Mark this FVG as processed
                            
            else:   
                if fvg_type == -1:
                    if self.data.Low[-1] <= fvg_bottom and fvg_bottom < self.ema[-1]:
                        entry_price = fvg_bottom
                        stop_loss = fvg_top * (1 + self.sl_offset)
                        take_profit = entry_price - ((stop_loss - entry_price) * self.risk_reward_ratio)
                        
                        if stop_loss > entry_price > take_profit:
                            self.sell(sl=stop_loss, tp=take_profit, size=0.1, limit=entry_price)
                            self.processed_fvgs.add(fvg_index)  # Mark this FVG as processed
                        
            
            
df = df.loc['2023-01-01' : '2024-01-01']              

bt = Backtest(df,
              FVGStrategy,
              cash=20000000,
              commission=.002,
              trade_on_close=True,
              exclusive_orders=False)  

# results = bt.optimize(
#     min_fvg_height=[10, 15, 20, 25, 30],
#     ema_period=[50, 75, 100, 125, 150, 175, 200],
#     sl_offset=[0.001,0.003, 0.005,0.007, 0.01],
#     risk_reward_ratio=[1.5, 2.0, 2.5, 3.0],
#     maximize='Return [%]'
# )



# # Print the optimized parameters
# print(f"Optimized parameters:")
# print(f"Best FVG height: {results._strategy.min_fvg_height}")
# print(f"Best EMA period: {results._strategy.ema_period}")
# print(f"Best stop loss offset: {results._strategy.sl_offset:.4f}")
# print(f"Best risk-reward ratio: {results._strategy.risk_reward_ratio:.2f}")
stats = bt.run()
print(stats)
# print(results)
bt.plot()