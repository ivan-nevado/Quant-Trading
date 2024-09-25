import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\TFG\BTC-6h-100wks-data.csv')

# Parse the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.loc['2023-01-01':'2024-01-01'].copy()

# Function to calculate pivot points (OB candidates)
def pivotid(df1, l, n1, n2): 
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0
    
    pividlow = True
    pividhigh = True
    for i in range(l - n1, l + n2 + 1):
        if df1['Low'].iloc[l] > df1['Low'].iloc[i]:
            pividlow = False
        if df1['High'].iloc[l] < df1['High'].iloc[i]:
            pividhigh = False
    if pividlow:
        return 1  # Bullish OB candidate
    elif pividhigh:
        return -1  # Bearish OB candidate
    else:
        return 0

# Function to detect OBs
def identify_ob(df):
    ob_signals = np.full(len(df), np.nan)
    top = np.full(len(df), np.nan)
    bottom = np.full(len(df), np.nan)
    mitigated_index = np.full(len(df), np.nan)
    ob_indices = np.full(len(df), np.nan)

    def is_inside_bar(prev_ob_index, new_ob_index, df):
        return df['High'][prev_ob_index] >= df['High'][new_ob_index] and df['Low'][prev_ob_index] <= df['Low'][new_ob_index]

    def is_pullback(ob_index, mit_index, df, ob_type):
        for j in range(ob_index + 1, mit_index):
            if ob_type == 1 and df['High'][j] < df['High'][j-1]:
                return True
            elif ob_type == -1 and df['Low'][j] > df['Low'][j-1]:
                return True
        return False
    
    most_recent_ob_index = None  # Track the most recent OB
    
    for i in range(1, len(df) - 2):
        pivot = pivotid(df, i, 3, 3)

        if pivot == 1:  # Bullish OB
            if most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index, i, df):
                ob_signals[i] = 1
                top[i] = df['High'].iloc[i]
                bottom[i] = df['Low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i
                        
        if pivot == -1:  # Bearish OB
            if most_recent_ob_index is None or not is_inside_bar(most_recent_ob_index, i, df):
                ob_signals[i] = -1
                top[i] = df['High'].iloc[i]
                bottom[i] = df['Low'].iloc[i]
                ob_indices[i] = i
                most_recent_ob_index = i

    for i in np.where(~np.isnan(ob_signals))[0]:
        ob_type = ob_signals[i]
        if ob_type == 1:  # Bullish OB
            for j in range(i + 1, len(df)):
                if df['Low'].iloc[j] <= top[i]:
                    if is_pullback(i, j, df, ob_type):
                        mitigated_index[i] = j
                        break
        elif ob_type == -1:  # Bearish OB
            for j in range(i + 1, len(df)):
                if df['High'].iloc[j] >= bottom[i]:
                    if is_pullback(i, j, df, ob_type):
                        mitigated_index[i] = j
                        break

    return pd.DataFrame({
        'OB': ob_signals,
        'Top': top,
        'Bottom': bottom,
        'MitigatedIndex': mitigated_index,
        'OBIndex': ob_indices
    })

# Function to calculate EMA
def calculate_ema(df, period=100):
    return df['Close'].ewm(span=period, adjust=False).mean()

# Apply OB detection and EMA calculation
ob_df = identify_ob(df)
df['EMA'] = calculate_ema(df)

# Function to plot order blocks and EMA using Plotly
def plot_order_blocks(df, ob_df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlesticks'
    )])

    # Add EMA to the plot
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='EMA 100'
    ))
    
    # Add OB rectangles for bullish and bearish order blocks
    for i in range(len(ob_df)):
        if not np.isnan(ob_df['OB'].iloc[i]):
            ob_type = ob_df['OB'].iloc[i]
            start_date = df.index[i]
            mitigation_index = int(ob_df['MitigatedIndex'].iloc[i]) if not np.isnan(ob_df['MitigatedIndex'].iloc[i]) else len(df) - 1
            end_date = df.index[mitigation_index]

            # Draw the rectangle for the OB
            fig.add_shape(type="rect",
                          x0=start_date, y0=ob_df['Bottom'].iloc[i],
                          x1=end_date, y1=ob_df['Top'].iloc[i],
                          fillcolor="rgba(0, 255, 0, 0.3)" if ob_type == 1 else "rgba(255, 0, 0, 0.3)",
                          line=dict(color="rgba(0, 255, 0, 0.3)" if ob_type == 1 else "rgba(255, 0, 0, 0.3)"))

    # Customize the layout
    fig.update_layout(
        title="Candlestick Chart with Order Blocks and EMA",
        xaxis_title="Date",
        yaxis_title="Price",
        width=1600,
        height=700
    )

    fig.show()

# Plot the data
plot_order_blocks(df, ob_df)
