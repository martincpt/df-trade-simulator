# Data Frame Trade Simulator
Simulates trading strategies in a Pandas data frame.

## Usage
```python
import pandas as pd

from df_trade_simulator import MarketDFTradeSimulator

# assuming you have a column with the name `price`
df = pd.read_csv("price_stream.csv") 

# create the instance
trade_sim = MarketDFTradeSimulator(df)

# add sides with value based selection
trade_sim.add_sides(buy=(df.price < 100), sell=(df.price > 200))

# run the simulation
trade_sim.simulate()

# print the trades only
print(trade_sim.df_trades)
```