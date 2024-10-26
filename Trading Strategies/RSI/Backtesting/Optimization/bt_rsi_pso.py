import pandas as pd
from backtesting import Backtest, Strategy
from pyswarm import pso
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

# List to store the best fitness value at each generation
best_fitness_per_gen = []

# Initialize variables to store the best position and fitness across generations
global_best_fitness = float('inf')  # Best fitness (lower is better)
global_best_position = None  # Best parameter set

# Define the fitness function for PSO with weights inside the function
def pso_fitness_function(params):
    ema_period, rsi_period, overbought, oversold = params
    results = bt.run(ema_period=int(ema_period), rsi_period=int(rsi_period), overbought=int(overbought), oversold=int(oversold))

    # Define weights for return and drawdown (customize these based on your preferences)
    weight_return = 1.0
    weight_drawdown = 1.0

    # Minimize a combination of negative return and drawdown
    return -weight_return * results['Return [%]'] + weight_drawdown * results['Max. Drawdown [%]']

# Define bounds for each parameter
lb = [50, 5, 70, 10]  # Lower bounds
ub = [200, 20, 90, 30]  # Upper bounds

# Set the number of generations and population size
num_generations = 100  # Customize as needed
population_size = 50  # Customize as needed

# Run PSO and manually track the best fitness at each generation
for gen in range(num_generations):
    print(f"Generation {gen + 1} running...")

    # Perform PSO for this generation
    xopt, fopt = pso(pso_fitness_function, lb, ub, maxiter=1, swarmsize=population_size)

    # If the new fitness is better than the global best, update global best
    if fopt < global_best_fitness:
        global_best_fitness = fopt
        global_best_position = xopt

    # Record the best fitness value for this generation
    best_fitness_per_gen.append(global_best_fitness)

    print(f"Best fitness at generation {gen + 1}: {global_best_fitness}")

# Plot the fitness improvement over generations
plt.plot(range(1, len(best_fitness_per_gen) + 1), best_fitness_per_gen, marker='o')
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.title('Fitness Improvement Over Generations with Elitism')
plt.grid(True)
plt.show()

# Use global_best_position for the final run with the optimized parameters
stats = bt.run(
    ema_period=int(global_best_position[0]), 
    rsi_period=int(global_best_position[1]), 
    overbought=int(global_best_position[2]), 
    oversold=int(global_best_position[3])
)

print(stats)
# bt.plot(resample=None)