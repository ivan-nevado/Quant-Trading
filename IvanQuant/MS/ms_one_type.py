import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load and preprocess data

dfpl = pd.read_csv(r"C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\IvanQuant\MS\BTC-USD (1).csv")

# Parse datetime
df = dfpl.copy()
df['Local time'] = pd.to_datetime(df['Local time'], format='%Y-%m-%d', utc=True)
df.rename(columns={'Local time': 'Date'}, inplace=True)

df = df[0:200]

# Function to find the first valid range based on dynamic high/low tracking
def find_first_range(ohlc: pd.DataFrame, min_swing_size: float):
    first_high = ohlc['high'].iloc[0]
    first_low = ohlc['low'].iloc[0]
    high_index = 0
    low_index = 0

    for i in range(1, len(ohlc)):
        current_high = ohlc['high'].iloc[i]
        current_low = ohlc['low'].iloc[i]

        if current_high > first_high:
            first_high = current_high
            high_index = i
        if current_low < first_low:
            first_low = current_low
            low_index = i

        if abs(first_high - first_low) >= min_swing_size:
            if current_high > ohlc['high'].iloc[low_index]:
                return low_index, i, "bullish"
            elif current_low < ohlc['low'].iloc[high_index]:
                return high_index, i, "bearish"

    return None, None, None

def retracements(end_level, swing_pivot, possible_pivot, min_retracement):  # Ensure passing values not indexes
    # Ensure that the denominator (end_level - swing_pivot) is not zero
    if end_level == swing_pivot or swing_pivot == possible_pivot:
        return False  # No retracement is possible if the swing pivot and end level are the same
    
    # Calculate retracement ratio, ensuring no division by zero
    retracement_ratio = (end_level - possible_pivot) / (end_level - swing_pivot)

    # Check if retracement meets the minimum retracement level
    if retracement_ratio >= min_retracement:
        return True
    else:
        return False

# Function to plot market structure and handle BOS logic
def plot_market_structure(ohlc: pd.DataFrame, min_swing_size: float):
    fig = go.Figure(data=[go.Candlestick(
         x=ohlc['Date'],
        open=ohlc['open'],
         high=ohlc['high'],
         low=ohlc['low'],
         close=ohlc['close']
     )])

   
    # Find the first valid range
    pivot_index, end_index, range_type = find_first_range(ohlc, min_swing_size)
    current_end = end_index
    current_pivot = pivot_index
    tracking_pivot = end_index
    sweep_pivot = 0

    # Continue plotting and finding BOS after the first range is identified
    for i in range(end_index + 1, len(ohlc)):
        if range_type == "bullish":
            # Price closing above previous high with valid retracement
            if ohlc['close'].iloc[i] > ohlc['high'].iloc[current_end] and current_pivot > current_end and retracements(ohlc['high'].iloc[current_end], ohlc['low'].iloc[pivot_index], ohlc['low'].iloc[current_pivot], min_retracement=0.382):
                print(f"Adding bullish line: pivot_index {pivot_index}, current_end {current_end}")
                print(f"x0: {ohlc['Date'].iloc[current_end]}, y0: {ohlc['high'].iloc[current_end]}")
                print(f"x1: {ohlc['Date'].iloc[i]}, y1: {ohlc['high'].iloc[current_end]}")
                fig.add_shape(
                    type="line",
                    x0=ohlc['Date'].iloc[current_end], y0=ohlc['high'].iloc[current_end],
                    x1=ohlc['Date'].iloc[i], y1=ohlc['high'].iloc[current_end],
                    line=dict(color="green", width=2)
                )
                pivot_index = current_pivot
                current_end = i
                tracking_pivot = i
                sweep_pivot = current_pivot
                
            # Update the current end if price moves higher without pullback
            elif ohlc['high'].iloc[i] > ohlc['high'].iloc[current_end] and current_pivot == pivot_index:
                current_end = i
                tracking_pivot = i
            elif ohlc['high'].iloc[i] > ohlc['high'].iloc[current_end] and current_pivot < current_end and not retracements(ohlc['high'].iloc[current_end], ohlc['low'].iloc[pivot_index], ohlc['low'].iloc[current_pivot], min_retracement=0.382):
                current_end = i
                tracking_pivot = i
            elif ohlc['high'].iloc[i] > ohlc['high'].iloc[current_end] :
                current_end = i
                tracking_pivot = i
            elif ohlc['high'].iloc[i] > ohlc['high'].iloc[current_end] and current_pivot > current_end and retracements(ohlc['high'].iloc[current_end], ohlc['low'].iloc[pivot_index], ohlc['low'].iloc[current_pivot], min_retracement=0.382):
                if ohlc['high'].iloc[i] > sweep_pivot:
                    sweep_pivot = i
                
            # Switch to bearish BOS if price closes below previous low
            if ohlc['close'].iloc[i] < ohlc['low'].iloc[pivot_index]:
                print(f"Price closed below the previous pivot low at pivot_index {pivot_index}, transitioning to bearish BOS")
                fig.add_shape(
                    type="line",
                    x0=ohlc['Date'].iloc[pivot_index], y0=ohlc['low'].iloc[pivot_index],
                    x1=ohlc['Date'].iloc[i], y1=ohlc['low'].iloc[pivot_index],
                    line=dict(color="red", width=2)
                )
                pivot_index = current_end
                tracking_pivot = i
                current_pivot = pivot_index
                current_end = i
                range_type = "bearish"
                sweep_pivot = current_pivot
                print(f"Switched to bearish mode, new pivot_index: {pivot_index}, new current_end: {current_end}")
            else: 
                # Track the low for potential pivot updates
                if ohlc['low'].iloc[i] < ohlc['low'].iloc[tracking_pivot]:
                    tracking_pivot = i
                    current_pivot = i


        elif range_type == "bearish":
            # Price closing below previous low with valid retracement
            if ohlc['close'].iloc[i] < ohlc['low'].iloc[current_end] and current_pivot > current_end and retracements(ohlc['low'].iloc[current_end], ohlc['high'].iloc[pivot_index], ohlc['high'].iloc[current_pivot], min_retracement=0.382):
                print(f"Adding bearish line: pivot_index {pivot_index}, current_end {current_end}")
                print(f"x0: {ohlc['Date'].iloc[current_end]}, y0: {ohlc['low'].iloc[current_end]}")
                print(f"x1: {ohlc['Date'].iloc[i]}, y1: {ohlc['low'].iloc[current_end]}")
                fig.add_shape(
                    type="line",
                    x0=ohlc['Date'].iloc[current_end], y0=ohlc['low'].iloc[current_end],
                    x1=ohlc['Date'].iloc[i], y1=ohlc['low'].iloc[current_end],
                    line=dict(color="red", width=2)
                )
                pivot_index = current_pivot
                tracking_pivot = i
                current_end = i
                sweep_pivot = current_pivot

            # Update the current end if price moves lower without pullback
            elif ohlc['low'].iloc[i] < ohlc['low'].iloc[current_end] and current_pivot == pivot_index:
                current_end = i
                tracking_pivot = i
                
            elif ohlc['low'].iloc[i] < ohlc['low'].iloc[current_end] and not retracements(ohlc['low'].iloc[current_end], ohlc['high'].iloc[pivot_index], ohlc['high'].iloc[current_pivot], min_retracement=0.382):
                current_end = i
                tracking_pivot = i
            
            elif current_pivot > current_end and retracements(ohlc['low'].iloc[current_end], ohlc['high'].iloc[pivot_index], ohlc['high'].iloc[current_pivot], min_retracement=0.382):
                if ohlc['low'].iloc[i] < sweep_pivot:
                    sweep_pivot = i

            # Switch to bullish BOS if price closes above previous high
            if ohlc['close'].iloc[i] > ohlc['high'].iloc[pivot_index]:
                print(f"Price closed above the previous pivot high at pivot_index {pivot_index}, transitioning to bullish BOS")
                fig.add_shape(
                    type="line",
                    x0=ohlc['Date'].iloc[pivot_index], y0=ohlc['high'].iloc[pivot_index],
                    x1=ohlc['Date'].iloc[i], y1=ohlc['high'].iloc[pivot_index],
                    line=dict(color="green", width=2)
                )
                
                pivot_index = current_end
                tracking_pivot = i
                current_pivot = pivot_index
                current_end = i
                range_type = "bullish"
                sweep_pivot = current_pivot
                
                print(f"Switched to bullish mode, new pivot_index: {pivot_index}, new current_end: {current_end}")
            else:
                # Track the high for potential pivot updates
                if ohlc['high'].iloc[i] > ohlc['high'].iloc[tracking_pivot]:
                    tracking_pivot = i
                    current_pivot = i

    fig.update_layout(
        title="Refined Market Structure with BOS and Swing Points",
        xaxis_title="Date",
        yaxis_title="Price",
        width=1600,
        height=800,
    )
    fig.show()

# Plot market structure with refined BOS logic
plot_market_structure(df, min_swing_size=0.04)
