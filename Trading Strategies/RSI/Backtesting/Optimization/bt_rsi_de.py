import pandas as pd
from backtesting import Backtest, Strategy
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Iván Quant Trading\TFG\BTC-6h-300wks-data.csv')

# Ensure 'Date' column exists and is properly formatted
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Function to calculate RSI (handling the backtesting.py internal data type)
def RSI(arr, period):
    series = pd.Series(arr)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Strategy Class
class EMARSI_Strategy(Strategy):
    ema_period = 150
    rsi_period = 10
    overbought = 70
    oversold = 30

    def init(self):
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, min_periods=self.ema_period).mean(), self.data.Close)
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

    def next(self):
        ema_value = self.ema[-1]
        rsi_value = self.rsi[-1]
        account_balance = self.equity
        risk_amount = account_balance * 0.01
        macro_tendency_bullish = self.data.Close[-1] > ema_value

        if macro_tendency_bullish and rsi_value < self.oversold:
            if not self.position:
                take_profit = self.data.Close[-1] * 1.03
                stop_loss = self.ema[-1] * 0.99
                price = self.data.Close[-1]
                risk_per_share = price - stop_loss
                position_size = min(risk_amount / risk_per_share, self.equity / price)
                position_size = max(1, round(position_size))
                self.buy(tp=take_profit, sl=stop_loss, size=position_size)

        elif not macro_tendency_bullish and rsi_value > self.overbought:
            if self.position:
                take_profit = self.data.Close[-1] * 0.97      
                stop_loss = self.ema[-1] * 1.01
                price = self.data.Close[-1]
                risk_per_share = stop_loss - price
                position_size = min(risk_amount / risk_per_share, self.equity / price)
                position_size = max(1, round(position_size))

df = df.loc['2019-01-01':'2024-01-01']

bt = Backtest(
    df, 
    EMARSI_Strategy, 
    cash=2000000, 
    commission=.002,
    trade_on_close=True,
    exclusive_orders=True
)

def objective_function(params):
    ema_period, rsi_period, overbought, oversold = params
    results = bt.run(ema_period=int(ema_period), rsi_period=int(rsi_period), overbought=int(overbought), oversold=int(oversold))

    # Set weights for return and drawdown
    weight_return = 1.0  # Weight for maximizing return (can be adjusted)
    weight_drawdown = 1.0  # Weight for minimizing drawdown (can be adjusted)

    # Combine the objectives into a single metric to minimize
    return -weight_return * results['Return [%]'] + weight_drawdown * results['Max. Drawdown [%]']

# Define bounds for each parameter
bounds = [(50, 200), (5, 20), (70, 90), (10, 30)]  # Bounds for ema_period, rsi_period, overbought, oversold

# Track the value of the objective function at each generation
fitness_values = []


# Callback function to store the value at each generation
def record_fitness(xk, convergence):
    value = objective_function(xk)
    fitness_values.append(value)
    print(f"Generation {len(fitness_values)}: Objective Function Value = {value}")

# Set the number of generations and population size
num_generations = 100  # Customize as needed
population_size = 50  # Customize as needed

# Run Differential Evolution
result = differential_evolution(
    objective_function, 
    bounds, 
    maxiter=num_generations, 
    popsize=population_size, 
    callback=record_fitness
)

# Use optimized parameters for the final run
stats = bt.run(
    ema_period=int(result.x[0]), 
    rsi_period=int(result.x[1]), 
    overbought=int(result.x[2]), 
    oversold=int(result.x[3])
)

print(f"Optimized parameters: EMA Period: {result.x[0]}, RSI Period: {result.x[1]}, Overbought: {result.x[2]}, Oversold: {result.x[3]}")
print(stats)

# Plot fitness values across generations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fitness_values) + 1), fitness_values, marker='o')
plt.title('DE Fitness Progression')
plt.xlabel('Generation')
plt.ylabel('Objective Function Value')
plt.grid(True)
plt.show()
# bt.plot(resample=None)