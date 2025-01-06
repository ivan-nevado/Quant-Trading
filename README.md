<h1 align="center">⚡ QUANTITATIVE TRADING STRATEGIES ⚡</h1>
<p align="center"><img src="https://img.shields.io/github/license/ivan-nevado/Quant-Trading" alt="LICENSE" /> </p>
<img src="https://img.shields.io/github/stars/ivan-nevado/Quant-Trading" alt="Stars" /> </p>
<h2>🚀 Project Overview</h2>
Overview</h2> This repository provides quantitative trading strategies using algorithms like big liquidation detection, funding arbitrage, strategies using technical analysis and machine learning models. 

<h2>🛠️ Key Strategies</h2>

<h3>1. Big Liquidations Detection</h3>

  - Detects large liquidation events in markets to signal significant price movements. <br>
  - **Usage** -> ```bash python big_liqs.py ```
<h3>2. Funding Rate</h3>
  - Tracks funding rates to identify arbitrage opportunities across exchanges.  <br>
  - **Usage** ->  ```bash python funding.py ``` 
<h3>3. Tracking Whale Orders</h3>
  - Monitors significant trades to identify institutional activity.
<h3>4. Orderblock Strategy</h3>
  Identify pivot zones where institutionals left a footprint when they entered and try to get involved when is likely that they will do too.
  **Features**: 
  - Indicator: Detects order blocks where price is likely to react. <br>
  - Backtesting: Backtest the strategy on historical data.<br>
  **Bots**: <br>
  - Centralized Exchange Bot: Automated bot for centralized exchanges (Phemex). <br>
  - Decentralized Exchange Bot: Automated bot for decentralized exchanges (Hyperliquid).

<h3>5. FairValueGap Strategy</h3>
  Detects imbalance between Supply and Demand and try to get involved when either Demand or Supply is in control.
   **Features**: 
   - Indicator: Identifies Fair Value Gaps to optimize entry/exit points. 
   - Backtesting: Test the strategy on historical data. 
   **Bots**: 
   - Centralized Exchange Bot: Automated bot for centralized exchanges (Phemex). <br>
   - Decentralized Exchange Bot: Automated bot for decentralized exchanges (Hyperliquid).
<h3>6. MultitimeFrame Analysis Strategy</h3>
  Use two timeframes to have a more accurate strategy and more precise entries
 - Backtesting: Test the strategy on historical data. <br>
 - Centralized Exchange Bot: Automated bot for centralized exchanges (Phemex). <br>
 - Decentralized Exchange Bot: Automated bot for decentralized exchanges (Hyperliquid).

<h2>🔄 Optimization Algorithms</h2>
   Implements 3 algorithms to improve the results by maximizing the profits and minimizing the drawdowns
<h3>RSI Strategy</h3>
Detects in what tendency we are trading within by the EMA and uses the RSI to enter in the market
<h4>🧬Genetic Algorithm</h4>
-Explanation:
A Genetic Algorithm is a search heuristic that mimics the process of natural selection. It is commonly used to solve optimization problems by iteratively improving a candidate solution based on survival of the fittest.
It evolves a population of potential solutions using selection, crossover, and mutation operations.
Key Concepts:
Population: A set of possible solutions.
Fitness Function: A function that evaluates how good a solution is (in this case, maximizing return and minimizing drawdown).
Selection: Choosing the best solutions to pass on their genes (parameters).
Crossover: Combining two solutions to create new offspring.
Mutation: Randomly altering a solution to explore new possibilities.
<h4>🧬Differential Evolution</h4>
-Explanation:
Differential Evolution is a population-based optimization algorithm that optimizes a problem by iteratively improving candidate solutions with respect to a given measure of quality (fitness function).
It works by adding weighted differences between solutions to create new solutions.
Key Concepts:
Mutation: In DE, the mutation is done by adding the weighted difference between two individuals to a third one.
Crossover: Combining mutated individuals with current solutions.
Selection: Keeping the better solution between the current and mutated ones.
<h4>🧬Particle Swarm Optimization</h4>
-Explanation:
Particle Swarm Optimization is a computational method inspired by the social behavior of birds. It searches for the optimal solution by having a swarm of particles (potential solutions) move around the solution space.
Particles adjust their positions based on their own experience and the experience of their neighbors.
Key Concepts:
Particles: Each particle represents a potential solution.
Velocity and Position Updates: Particles move through the solution space by updating their positions and velocities based on personal and swarm best experiences.
<h2>🤖 Machine Learning Algorithms</h2>
Implemented ML algorithms in QuantConnect using different models:
<h3>Linear Regression + PCA</h3>
- PCA (Principal Component Analysis)🧠💻: PCA is a machine learning model used for dimensionality
  reduction.
  This simplifies the input features while retaining the most important information.<br>
- Linear Regression📉📈: Linear Regression is another machine learning model used to predict future returns
  based on the transformed data.
  It models the relationship between the
  principal components and future returns in a linear way.
<h3>Decision Tree🌳🔍</h3>
- Decision Tree Classifier: A decision tree is a supervised learning algorithm that splits data into branches
  based on the values of input features. Each branch represents a decision, and the model selects the best
  features to split the data, helping to predict an outcome.
<h3>Support Vector Machine🔀🤖</h3>
- The SVM classifies whether the market is likely to be bullish (prices increasing) or bearish (prices
  decreasing) based on SPY’s technical indicators.<br>
- Kernel Trick: The SVM uses the RBF (Radial Basis Function) kernel to handle non-linear relationships
  between indicators, making it well-suited for capturing complex market behavior.
<h3>Recurrent Neuronal Network🔄🧠</h3>
- Recurrent Neural Networks (RNNs): Unlike traditional feedforward neural networks, RNNs have loops in
  their architecture that allow them to maintain a “memory” of past inputs. This makes RNNs particularly
  well-suited for tasks involving sequential data, such as time series predictions or stock price forecasting.<br>
- Stock prices form a time series, meaning that the order in which data points (prices, volumes)
  occur is important. RNNs can remember and capture dependencies in this sequence, allowing the model
  to learn meaningful patterns over time.
<h3>LSTM📊🔁🧠</h3>
- Long Short-Term Memory (LSTM): An advanced type of Recurrent Neural Network (RNN) designed to
  capture both short-term and long-term dependencies in sequential data like stock prices.
