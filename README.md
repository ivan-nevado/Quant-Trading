<h1 align="center">⚡ QUANTITATIVE TRADING STRATEGIES ⚡</h1>
<p align="center"><img src="https://img.shields.io/github/license/ivan-nevado/Quant-Trading" alt="License" /> </p>
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


   

    
