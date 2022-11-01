# Data Frame Trade Simulator
Simulates trading strategies through out a Pandas data frame.

![Requirement: Python >= 3.10](https://img.shields.io/badge/Python-%3E%3D%203.10-blue)

## Quickstart

```python
import pandas as pd

from df_trade_simulator import MarketDFTradeSimulator

# assuming you have a column with the name `price`
df = pd.read_csv("price_stream.csv") 

# create the instance
trade_sim = MarketDFTradeSimulator(df)

# add signals by value based selections
trade_sim.add_signals(buy=(df.price < 100), sell=(df.price > 200))

# run the simulation
trade_sim.simulate()

# print the trades only
print(trade_sim.df_trades)
```

## Usage

Assuming you have a table with **a price column** and bunch of bad-ass indicators (optional). Price column is required so the trade simulator can calculate the ROI (Return of Investment) and Wallet status.

The simulator will extend your data frame with the following columns, so please be assure they are safe to overwrite:
- **signal**: Literal["buy", "sell"]
- **roi**: float
- **wallet**: float
- **wallet_fee**: float

## Initialize

```python
# create the instance
trade_sim = MarketDFTradeSimulator(df)

# in case you call your price column differently
trade_sim = MarketDFTradeSimulator(df, price_col="close")

# you can also change the other pre-defined column names
trade_sim = MarketDFTradeSimulator(
    df, 
    price_col="close",
    signal_col="side",
    wallet_col="total",
    wallet_fee_col="taxed",
    roi_col="profit",
)
```

## Add signals

`.add_signals` uses pandas' `.loc` method so it accepts the same inputs as you would [access a group of rows and columns by label(s) or a boolean array](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html).

```python
# simple selection
trade_sim.add_signals(buy=(df.price < 100), sell=(df.price > 200))

# complex selection with bitwise operators
trade_sim.add_signals(
    buy=((df.price < 100) | (df.badass_indicator == "buy")), 
    sell=((df.price > 200) & (df.is_ath)),
)
```

*Adding signals is optional. You can have a data frame with pre-defined signals but that must contain labels literally as `buy` and `sell`.*

## Available Trading Strategies
- **Market**: MarketDFTradeSimulator
  - **Sell**: Sells everything when receiving the signal 
  - **Buy**: Buys as much as it can when receiving the signal
- **Stop-Limit**: StopLimitDFTradeSimulator
  - Same as **Market** strategy but has an active stop limit on all trades. Stop-limit must be set via the `treshold` argument, which is a percentage value indicating how much you are willing to loose before quitting the position.

```python
from df_trade_simulator import MarketDFTradeSimulator
from df_trade_simulator import StopLimitDFTradeSimulator 

# Regular trade strategy
trade_sim = MarketDFTradeSimulator(df)

# Stop-Limit trade strategy
trade_sim = StopLimitDFTradeSimulator(df, treshold=0.1)
```