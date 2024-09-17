import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load the data
df = pd.read_csv(r"C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\Indicators\OrderBlock\EURUSD_Candlestick_1_D_ASK_05.05.2003-19.10.2019 - copia.csv")
df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
df = df[df['volume'] != 0]  # Filter out rows with zero volume
df.reset_index(drop=True, inplace=True)
# df = df.loc['2022-01-01':'2023-01-01']

# Pivot point calculation logic
def pivotid(df1, l, n1, n2): 
    # Check boundaries (avoid going out of the dataframe)
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0
    
    pividlow = True
    pividhigh = True
    # Checking local highs and lows within the given window (n1, n2)
    for i in range(l - n1, l + n2 + 1):
        if df1.low[l] > df1.low[i]:
            pividlow = False
        if df1.high[l] < df1.high[i]:
            pividhigh = False
    if pividlow:
        return 1  # Pivot Low (Bullish OB Candidate)
    elif pividhigh:
        return 2  # Pivot High (Bearish OB Candidate)
    else:
        return 0

# Apply the pivot detection
df['pivot'] = df.apply(lambda x: pivotid(df, x.name, 3, 3), axis=1)

# Detecting order blocks based on the pivot highs and lows
def detect_order_blocks(df):
    ob_type = pd.Series(np.nan, index=df.index)  # Will hold the OB type (1 for bullish, -1 for bearish)
    ob_top = pd.Series(np.nan, index=df.index)  # Upper boundary of the OB
    ob_bottom = pd.Series(np.nan, index=df.index)  # Lower boundary of the OB
    
    for i in range(1, len(df) - 1):
        if df['pivot'].iloc[i] == 1:  # Bullish OB
            ob_type[i] = 1  # Mark as bullish OB
            ob_top[i] = df['high'].iloc[i]  # OB top is the high of the pivot candle
            ob_bottom[i] = df['low'].iloc[i]  # OB bottom is the low of the pivot candle
            
        elif df['pivot'].iloc[i] == 2:  # Bearish OB
            ob_type[i] = -1  # Mark as bearish OB
            ob_top[i] = df['high'].iloc[i]  # OB top is the high of the pivot candle
            ob_bottom[i] = df['low'].iloc[i]  # OB bottom is the low of the pivot candle
    
    return pd.DataFrame({
        'OB_Type': ob_type,
        'OB_Top': ob_top,
        'OB_Bottom': ob_bottom
    })

# Apply the function to detect order blocks
order_blocks_df = detect_order_blocks(df)

# Function to determine if an Order Block is filled
def is_ob_filled(ob_row, df, start_index):
    ob_top = ob_row['OB_Top']
    ob_bottom = ob_row['OB_Bottom']
    
    for i in range(start_index, len(df)):
        if (ob_row['OB_Type'] == 1 and df['low'].iloc[i] <= ob_bottom):  # Bullish OB filled when candle touches bottom
            return i  # Return the index where it is filled
        elif (ob_row['OB_Type'] == -1 and df['high'].iloc[i] >= ob_top):  # Bearish OB filled when candle touches top
            return i  # Return the index where it is filled
    return None  # Return None if not filled

# Plotting the candlestick chart with OBs and filled detection
def plot_order_blocks(df, ob_df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    
    # Add OB rectangles for bullish and bearish order blocks
    for i in range(len(ob_df)):
        if not np.isnan(ob_df['OB_Type'].iloc[i]):
            start_date = i
            end_date = i + 1

            # Find the first index where the OB is filled
            filled_index = is_ob_filled(ob_df.iloc[i], df, i + 1)
            
            # If the OB is filled, update end_date to the filled_index
            if filled_index is not None:
                end_date = filled_index

            # If the end_date exceeds the length of the dataframe, cap it
            if end_date >= len(df):
                end_date = len(df) - 1

            # Draw the rectangle for the OB
            fig.add_shape(type="rect",
                          x0=start_date, y0=ob_df['OB_Bottom'].iloc[i],
                          x1=end_date, y1=ob_df['OB_Top'].iloc[i],
                          fillcolor="rgba(0, 255, 0, 0.3)" if ob_df['OB_Type'].iloc[i] == 1 else "rgba(255, 0, 0, 0.3)",
                          line=dict(color="rgba(0, 255, 0, 0.3)" if ob_df['OB_Type'].iloc[i] == 1 else "rgba(255, 0, 0, 0.3)"))

    fig.update_layout(title="Candlestick Chart with Order Blocks", xaxis_rangeslider_visible=False)
    fig.show()

# Example usage
df_filtered = df[0:100]  # Slice of the dataframe for plotting
order_blocks_filtered = order_blocks_df[0:100]
plot_order_blocks(df_filtered, order_blocks_filtered)


