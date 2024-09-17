'''This strat only trades between 9:30-10:30'''

import backtesting
import talib
import pandas as pd

class SMATradingStrategy(backtesting.Strategy):
    stop_loss_pct = 5
    take_profit_pct = 1

    def init(self):
        self.sma5 = self.I(talib.SMA, self.data.Close, timeperiod=5)
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.entry_hour = 9
        self.exit_hour = 10

    def next(self):
        # Ensure Datetime is available and correctly formatted
        try:
            hour = pd.Timestamp(self.data.index[-1]).to_pydatetime().hour
        except KeyError:
            raise ValueError("Datetime index is missing from data")

        if self.position:
            if hour >= self.exit_hour:
                self.position.close()

        else:
            if hour >= self.entry_hour and self.sma5[-1] < self.sma20[-1]:
                entry_price = self.data.Close[-1]
                stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
                take_profit = entry_price * (1 + self.take_profit_pct / 100)

                if stop_loss < entry_price < take_profit:
                    self.buy(
                        size=1,
                        sl=stop_loss,
                        tp=take_profit
                    )
                else:
                    print(f"Skipping order due to SL/TP issue: SL={stop_loss}, Entry={entry_price}, TP={take_profit}")

# Test the strategy and pull in the data
data = pd.read_csv(r'C:\Users\Iván\Desktop\Ivan\Algorithmic_Trading\Day 4\Algo_Orders\BTC-15m-100wks-data.csv')

# Ensure column names are correctly capitalized
data.columns = [column.capitalize() for column in data.columns]

# Drop rows with missing values
data = data.dropna()

# Ensure 'Datetime' column is in datetime format
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Set 'Datetime' as index and ensure it's sorted in ascending order
data.set_index('Datetime', inplace=True)
data.sort_index(inplace=True)

# Run the backtest
bt = backtesting.Backtest(data, SMATradingStrategy, cash=1000000, commission=.002)
result = bt.optimize(
    maximize='Equity Final [$]',
    stop_loss_pct=range(5, 20),
    take_profit_pct=range(1, 10)
)

print(result)
bt.plot()
