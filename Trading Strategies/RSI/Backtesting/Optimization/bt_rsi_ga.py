'''This bot will look at the ema of 150 periods and see if we are bullish or bearish
If we are bullish and on the rsi we are oversold(below the 30 value) we buy
If we are bearish and on the rsi we are overbought(above the 70 value) we sell'''

import pandas as pd
from backtesting import Backtest, Strategy
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# Load the BTCUSD data from the CSV file
df = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Iván Quant Trading\TFG\BTC-6h-300wks-data.csv')

# Ensure 'Date' column exists and is properly formatted
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure the DataFrame has the required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

def RSI(arr, period):
    if period <= 0:
        raise ValueError("The period must be a positive integer.")
    
    series = pd.Series(arr)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Strategy Class
class EMARSI_Strategy(Strategy):
    ema_period = 150  # EMA period for trend detection
    rsi_period = 10  # RSI period for overbought/oversold
    overbought = 70  # Overbought threshold for RSI
    oversold = 30  # Oversold threshold for RSI

    def init(self):
        # Register the 10-period EMA to assess trend direction
        self.ema = self.I(lambda x: pd.Series(x).ewm(span=self.ema_period, min_periods=self.ema_period).mean(), self.data.Close)
        
        # Register the RSI indicator
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

    def next(self):
        # Get the most recent EMA and RSI values
        ema_value = self.ema[-1]
        rsi_value = self.rsi[-1]
        account_balance = self.equity
        risk_amount = account_balance * 0.01
        # Determine the macro tendency: bullish if price is above EMA, bearish if below EMA
        macro_tendency_bullish = self.data.Close[-1] > ema_value

        # Trading logic based on EMA and RSI:
        # Buy if bullish trend and RSI indicates oversold (RSI < 30)
        if macro_tendency_bullish and rsi_value < self.oversold:
            if not self.position:
                take_profit = self.data.Close[-1] * 1.03
                stop_loss = self.ema[-1] * 0.99
                price = self.data.Close[-1]
                risk_per_share = price - stop_loss
                position_size = min(risk_amount / risk_per_share, self.equity / price)
                position_size = max(1, round(position_size))  # Ensure at least 1 unit
                self.buy(tp=take_profit, sl=stop_loss, size=position_size)
                

        # Sell if bearish trend and RSI indicates overbought (RSI > 70)
        elif not macro_tendency_bullish and rsi_value > self.overbought:
            if self.position:
                take_profit = self.data.Close[-1] * 0.97      
                stop_loss = self.ema[-1] * 1.01
                price = self.data.Close[-1]
                risk_per_share = stop_loss - price
                position_size = min(risk_amount / risk_per_share, self.equity / price)
                position_size = max(1, round(position_size))  # Ensure at least 1 unit
                self.sell(tp=take_profit, sl=stop_loss, size=position_size)
# Filter the dataset for a specific date range
df = df.loc['2019-01-01':'2024-01-01']

# Perform the backtest using BTCUSD data
bt = Backtest(
    df, 
    EMARSI_Strategy, 
    cash=2000000, 
    commission=.002,
    trade_on_close=True,
    exclusive_orders=True
)

# Define the weights for the return and drawdown combination
weight_return = 0.5  # Adjust this weight as needed
weight_drawdown = 0.5  # Adjust this weight as needed

# Track combined fitness for each generation
generation_combined_fitness = []
generation_drawdown = []
generation_return = []
def fitness_function(individual):
    ema_period, rsi_period, overbought, oversold = individual

    # Ensure ema_period and rsi_period are positive integers
    ema_period = max(1, int(ema_period))  # Set a minimum of 1 for EMA period
    rsi_period = max(1, int(rsi_period))   # Set a minimum of 1 for RSI period

    results = bt.run(
        ema_period=ema_period, 
        rsi_period=rsi_period, 
        overbought=int(overbought),     
        oversold=int(oversold)
    )
    
    # Calculate combined fitness
    combined_fitness = (-weight_return * results['Return [%]']) + (weight_drawdown * results['Max. Drawdown [%]'])

    # Return combined fitness as a tuple
    return (combined_fitness,)



# Define individual and population in DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Maximize return, minimize drawdown, maximize combined fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_ema_period", random.randint, 50, 200)
toolbox.register("attr_rsi_period", random.randint, 5, 20)
toolbox.register("attr_overbought", random.randint, 70, 90)
toolbox.register("attr_oversold", random.randint, 10, 30)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_ema_period, toolbox.attr_rsi_period, toolbox.attr_overbought, toolbox.attr_oversold), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def track_generations(population, toolbox, cxpb, mutpb, ngen):
    best_individual = None  # To store the best individual across generations

    for gen in range(ngen):
        elite = tools.selBest(population, 1)[0]
        population, _ = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1, verbose=False)
        population[0] = elite

        combined_fitness_values = [ind.fitness.values[0] for ind in population]  # Get combined fitness values
        best_combined_fitness = min(combined_fitness_values)  # Get the best combined fitness

        generation_combined_fitness.append(best_combined_fitness)  # Store best combined fitness

        print(f"Generation {gen+1}, Best Combined Fitness: {best_combined_fitness}")


# Run the genetic algorithm
population = toolbox.population(n=50)
track_generations(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=100)

# Plot drawdown vs generations
plt.figure(figsize=(15, 5))

# Plot combined fitness vs generations
plt.subplot(1, 3, 3)
plt.plot(range(1, len(generation_combined_fitness) + 1), generation_combined_fitness, marker='o', color='orange')
plt.title('Combined Fitness vs Generations')
plt.xlabel('Generation')
plt.ylabel('Combined Fitness')
plt.grid(True)

plt.tight_layout()
plt.show()
# bt.plot(resample=None)