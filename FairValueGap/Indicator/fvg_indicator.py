import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\BTC-6h-100wks-data.csv')

# Parse the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Filter the data range to one year (as an example)
df = df.loc['2022-01-01':'2022-12-31'].copy()

# Define Fair Value Gap detection function with MitigatedIndex
def fvg(ohlc: pd.DataFrame):
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

# Detect FVGs
fvg_df = fvg(df)

# Function to determine if an FVG is filled
'''To see if the fvg is filled it must look since the fvg index + 2 so on checking when it is filled. 
    Have to be careful to dont invalidate a fvg bc if filled just bc in the future it gets filled'''
def is_fvg_filled(fvg_row, df, start_index):
    top = fvg_row['Top']
    bottom = fvg_row['Bottom']
    for i in range(start_index, len(df)):
        if (fvg_row['FVG'] == 1 and df['Low'].iloc[i] < bottom) or \
           (fvg_row['FVG'] == -1 and df['High'].iloc[i] > top):
            return i  # Return the index where it is filled
        
    return None  # Return None if not filled

# Plotting with Plotly (Dynamic Rectangles)
def plot_fvg_with_rectangles(df, fvg_df, min_fvg_height):
    # df = df.iloc[250:400]
    # fvg_df = fvg_df.iloc[250:400]

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])

    for i in range(len(fvg_df)):
        fvg_index = fvg_df['FVGIndex'].iloc[i]
        print(f'fvg index: {fvg_index}')
        if not np.isnan(fvg_df['FVG'].iloc[i]):
            start_date = df.index[i]
            end_date = df.index[min(i + 1, len(df)-1)]
            mit_index = fvg_df['MitigatedIndex'].iloc[i]

            # Ensure FVG meets the minimum length requirement
            if i + 2 < len(df):  # To check if there's space for at least 3 bars
                min_index = i + 2
                if min_index < len(df):
                    # Calculate FVG length in bars
                    fvg_length = min_index - i + 1
                    
            filled_index = is_fvg_filled(fvg_df.iloc[i], df, i + 2)
            
            # If the FVG is filled, update end_date to the filled_index
            if filled_index is not None:
                end_date = df.index[filled_index]

            # If the end_date exceeds the length of the dataframe, cap it
            if filled_index and filled_index >= len(df):
                end_date = df.index[-1]

            # Check the height of the FVG
            fvg_height = fvg_df['Top'].iloc[i] - fvg_df['Bottom'].iloc[i]
            if fvg_height >= min_fvg_height:
                fvg_index = fvg_df['FVGIndex'].iloc[i]
                print(f'index of the fvg: {fvg_index}')
                print(f'The fvg was completely filled in this index: {filled_index}')
                print(f'the fvg was mitigated at index: {mit_index}')
                # Draw the rectangle for the FVG
                fig.add_shape(type="rect",
                              x0=start_date, y0=fvg_df['Bottom'].iloc[i],
                              x1=end_date, y1=fvg_df['Top'].iloc[i],
                              fillcolor="rgba(0, 255, 0, 0.3)" if fvg_df['FVG'].iloc[i] == 1 else "rgba(255, 0, 0, 0.3)",
                              line=dict(color="rgba(0, 255, 0, 0.3)" if fvg_df['FVG'].iloc[i] == 1 else "rgba(255, 0, 0, 0.3)"))
    
    fig.update_layout(
        title="Candlestick Chart with Fair Value Gaps",
        xaxis_title="Date",
        yaxis_title="Price",
        width=1600,  # Increase width
        height=800,  # Increase height
    )
    fig.show()

# Set minimum FVG height and length (adjust these values as needed)
min_fvg_height = 20

# Plot the results
plot_fvg_with_rectangles(df, fvg_df, min_fvg_height)
