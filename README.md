<h1 align="center">⚡ QUANTITATIVE TRADING STRATEGIES ⚡</h1>
<p align="center"><img src="https://img.shields.io/github/license/ivan-nevado/Quant-Trading" alt="LICENSE" /> </p>
<img src="https://img.shields.io/github/stars/ivan-nevado/Quant-Trading" alt="Stars" /> </p>
<h2>🚀 Project Overview</h2>
Overview</h2> This repository provides quantitative trading strategies using algorithms like big liquidation detection, Fair Value Gap (FVG) analysis, and funding arbitrage. 
The focus is on leveraging high-volume trades and market data to optimize performance.

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
