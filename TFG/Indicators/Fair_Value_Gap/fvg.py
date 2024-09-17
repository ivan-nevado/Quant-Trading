import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load and preprocess data
dfpl = pd.read_csv(r"C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\Indicators\Fair_Value_Gap\EURUSD_Candlestick_1_D_ASK_05.05.2003-19.10.2019 - copia (2).csv")
dfpl = dfpl[dfpl['volume'] != 0]
dfpl.reset_index(drop=True, inplace=True)

# Select a subset of the data
df = dfpl.loc[0:50].copy()

# Parse datetime correctly
df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z', utc=True)
df.rename(columns={'Local time': 'Date'}, inplace=True)

# Define Fair Value Gap detection function with MitigatedIndex
def fvg(ohlc: pd.DataFrame):
    fvg_signals = np.where(
        (
            (ohlc["high"].shift(1) < ohlc["low"].shift(-1))
            & (ohlc["close"] > ohlc["open"])
        )
        | (
            (ohlc["low"].shift(1) > ohlc["high"].shift(-1))
            & (ohlc["close"] < ohlc["open"])
        ),
        np.where(ohlc["close"] > ohlc["open"], 1, -1),
        np.nan,
    )
    
    top = np.where(
        ~np.isnan(fvg_signals),
        np.where(
            ohlc["close"] > ohlc["open"],
            ohlc["low"].shift(-1),
            ohlc["low"].shift(1),
        ),
        np.nan,
    )
    
    bottom = np.where(
        ~np.isnan(fvg_signals),
        np.where(
            ohlc["close"] > ohlc["open"],
            ohlc["high"].shift(1),
            ohlc["high"].shift(-1),
        ),
        np.nan,
    )
    
    # Calculate MitigatedIndex
    mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
    for i in np.where(~np.isnan(fvg_signals))[0]:
        mask = np.zeros(len(ohlc), dtype=np.bool_)
        if fvg_signals[i] == 1:
            mask = ohlc["low"][i + 2 :] <= top[i]
        elif fvg_signals[i] == -1:
            mask = ohlc["high"][i + 2 :] >= bottom[i]
        if np.any(mask):
            j = np.argmax(mask) + i + 2
            mitigated_index[i] = j

    mitigated_index = np.where(np.isnan(fvg_signals), np.nan, mitigated_index)
    
    return pd.concat(
        [
            pd.Series(fvg_signals, name="FVG"),
            pd.Series(top, name="Top"),
            pd.Series(bottom, name="Bottom"),
            pd.Series(mitigated_index, name="MitigatedIndex"),
        ],
        axis=1,
    )

# Detect FVGs
fvg_df = fvg(df)

# Function to determine if an FVG is filled
def is_fvg_filled(fvg_row, df, start_index):
    top = fvg_row['Top']
    bottom = fvg_row['Bottom']
    for i in range(start_index, len(df)):
        if (fvg_row['FVG'] == 1 and df['low'].iloc[i] <= bottom) or \
           (fvg_row['FVG'] == -1 and df['high'].iloc[i] >= top):
            return i  # Return the index where it is filled
    return None  # Return None if not filled

# Plotting with Plotly (Dynamic Rectangles)
def plot_fvg_with_rectangles(df, fvg_df, min_fvg_height):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])

    for i in range(len(fvg_df)):
        if not np.isnan(fvg_df['FVG'].iloc[i]):
            start_date = i
            end_date = i + 1

            # Ensure FVG meets the minimum length requirement
            if start_date + 2 < len(df):  # To check if there's space for at least 3 bars
                min_index = i + 2
                if min_index < len(df):
                    # Calculate FVG length in bars
                    fvg_length = min_index - start_date + 1
                    

            # Find the first index where the FVG is filled
            filled_index = is_fvg_filled(fvg_df.iloc[i], df, i + 3)
            
            # If the FVG is filled, update end_date to the filled_index
            if filled_index is not None:
                end_date = filled_index

            # If the end_date exceeds the length of the dataframe, cap it
            if end_date >= len(df):
                end_date = len(df) - 1

            # Check the height of the FVG
            fvg_height = fvg_df['Top'].iloc[i] - fvg_df['Bottom'].iloc[i]
            if fvg_height >= min_fvg_height:
                # Draw the rectangle for the FVG
                fig.add_shape(type="rect",
                              x0=start_date, y0=fvg_df['Bottom'].iloc[i],
                              x1=end_date, y1=fvg_df['Top'].iloc[i],
                              fillcolor="rgba(0, 255, 0, 0.3)" if fvg_df['FVG'].iloc[i] == 1 else "rgba(255, 0, 0, 0.3)",
                              line=dict(color="rgba(0, 255, 0, 0.3)" if fvg_df['FVG'].iloc[i] == 1 else "rgba(255, 0, 0, 0.3)"))

    fig.update_layout(title="Candlestick Chart with Fair Value Gaps", xaxis_title="Index", yaxis_title="Price")
    fig.show()

# Set minimum FVG height and length (adjust these values as needed)
min_fvg_height = 0.005

# Plot the results
plot_fvg_with_rectangles(df, fvg_df, min_fvg_height)
