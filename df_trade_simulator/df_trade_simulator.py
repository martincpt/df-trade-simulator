from dataclasses import dataclass
import pandas as pd

from pandas import DataFrame, Series

from abc import ABC, abstractmethod
from typing import Any, Literal


Side = Literal["buy", "sell"]

# Define constants
BUY: Side = Side.__args__[0]
SELL: Side = Side.__args__[1]
HODL: str = "hodl"

# Default columns
PRICE_COL = "price"
WALLET_COL = "wallet"
WALLET_FEE_COL = "wallet_fee"
ROI_COL = "roi"
SIDE_COL = "side"

# Other config values
INIT_WALLET = 1


@dataclass
class DFTrade:
    row: Series
    price: float
    side: Side
    roi: float
    wallet: float = INIT_WALLET
    wallet_fee: float = INIT_WALLET


class DFTradeSimulator(ABC):
    df: DataFrame
    # trades
    trades: list[DFTrade]
    # row states
    current_row: Series
    last_row: Series
    # helpers
    x_fee: float
    columns: tuple[str]

    def __init__(self, df: DataFrame, **kwargs) -> None:
        # read kwargs
        self.fee = kwargs.get("fee", 0.1)
        self.price_col = kwargs.get("price_col", PRICE_COL)
        self.wallet_col = kwargs.get("wallet_col", WALLET_COL)
        self.wallet_fee_col = kwargs.get("wallet_fee_col", WALLET_FEE_COL)
        self.roi_col = kwargs.get("roi_col", ROI_COL)
        self.side_col = kwargs.get("side_col", SIDE_COL)
        # create list of columns
        self.columns = self.side_col, self.roi_col, self.wallet_col, self.wallet_fee_col
        # set data frame
        self.set_df(df)
        # set fee
        self.x_fee = (100 - self.fee) / 100

    def set_df(self, df: DataFrame):
        """Sets the current data frame and cleans up existing trades."""
        # make a copy
        self.df = df.copy()
        # make sure they exist without override
        # NOTE: defining columns here not only unifies the output data frame
        # (because none of these column values added if no trade produced)
        # but also keeps the original order of columns which is important
        for col in self.columns:
            if col not in self.df:
                self.df[col] = float("NaN")
        # clean up
        self.trades = []

    def add_sides(self, buy: Series, sell: Series) -> None:
        """Adds sides by Series."""
        self.df.loc[buy, self.side_col] = BUY
        self.df.loc[sell, self.side_col] = SELL

    def simulate_event_pre(self, row: Series) -> None:
        """Runs at the beginning of each self.simulate_event().

        Responsible to store self.current_row.

        Can be used to prepare any individual event.

        Args:
            row (Series): The current row passed to self.simulate_event().
        """
        self.current_row = row

    def simulate_event_post(self, row: Series) -> None:
        """Runs at the end of each self.simulate_event().

        Responsible to store self.last_row.

        Can be used to close or clean up any individual event.

        Args:
            row (Series): The current row passed to self.simulate_event().
        """
        self.last_row = row

    def simulate_event(self, row: Series):
        """Simulates as an individual sample / data frame row comes in.
        This function is responsable to extend the row with ROI and
        wallet informations and return with it.

        Args:
            row (Series): The row to simulate.

        Returns:
            Series: The row extended with roi, wallet and wallet fee columns.
        """
        # pre process
        self.simulate_event_pre(row)

        if self.should_trade():
            trade = self.do_trade()
            self.trades.append(trade)
            self.extend_row_with_trade(row, trade)

        # post process
        self.simulate_event_post(row)
        # return
        return row

    def simulate(self) -> None:
        """Simulates the state of the wallet by going through the data frame.

        Extends the data frame with `roi`, `wallet` and `wallet_fee` columns by
        passing each row to self.simulate_event()."""
        # put HODL to last row so we will get a wallet balance
        # calculation at the very end
        self.df.iloc[-1, self.df.columns.get_loc(self.side_col)] = HODL

        self.df = self.df.apply(self.simulate_event, axis=1)

    @abstractmethod
    def should_trade(self) -> bool:
        """Determines whether a trade should be done."""

    def do_trade(self) -> DFTrade:
        """Handles trading procedure.

        Returns:
            DFTrade: Returns the processed trade.
        """

        # calc roi and wallet values
        roi = self.calc_roi()
        wallet = self.calc_wallet(roi)
        wallet_fee = self.calc_wallet(roi, fee=True)

        # get args
        args = {
            "row": self.current_row,
            "price": self.current_price,
            "side": self.signal,
            "roi": roi,
            "wallet": wallet,
            "wallet_fee": wallet_fee,
        }

        # create instance
        trade = DFTrade(**args)

        return trade

    def calc_roi(self) -> float:
        """Calculates ROI (Return of Investment) since last trade."""
        # get prices and side
        current_price = self.current_price
        last_price = self.last_price
        side = self.side
        # print(f"{side = } {current_price = } {last_price = }")
        # calc roi
        return current_price / last_price if side == BUY else last_price / current_price

    def calc_wallet(self, roi: float, fee: bool = False) -> float:
        """Calculates wallet balance since the last trade.

        Args:
            roi (float): ROI of last trade.
            fee (bool, optional): Whether to apply fee on it. Defaults to False.

        Returns:
            float: Wallet balance.
        """
        wallet = self.wallet * roi
        return wallet if not fee else wallet * self.x_fee

    def extend_row_with_trade(self, row: Series, trade: DFTrade) -> None:
        """Extends row with trade datas.

        Args:
            row (Series): The row to be extended.
            trade (DFTrade): The trade object to extend with.
        """
        row[self.roi_col] = trade.roi
        row[self.wallet_col] = trade.wallet
        row[self.wallet_fee_col] = trade.wallet_fee

    def nan_safe(self, value: Any) -> Any:
        """Replaces NaN with None."""
        return None if pd.isna(value) else value

    @property
    def last_trade(self) -> DFTrade | None:
        """Returns the last trade or None if no trade been found."""
        return self.trades[-1] if len(self.trades) > 0 else None

    @property
    def wallet(self) -> float:
        """Returns the current state of wallet."""
        last_trade = self.last_trade
        return INIT_WALLET if last_trade is None else last_trade.wallet

    @property
    def wallet_fee(self) -> float:
        """Returns the current state of wallet with fees applied."""
        last_trade = self.last_trade
        return INIT_WALLET if last_trade is None else last_trade.wallet_fee

    @property
    def side(self) -> Side | None:
        """Returns the side from last trade or None if no trade was found."""
        return None if self.last_trade is None else self.nan_safe(self.last_trade.side)

    @property
    def signal(self) -> Side | None:
        """Returns the market side to change to (a.k.a. signal) from self.current_row.

        Please note this property is only up to date once self.simulate_event_pre()
        has updated self.current_row. Otherwise it may return None or an
        outdated result.
        """
        row = self.current_row
        return None if row is None else self.nan_safe(row[self.side_col])

    @property
    def current_price(self) -> float | None:
        """Returns the price from the current row."""
        return getattr(self.current_row, self.price_col, None)

    @property
    def last_price(self) -> float | None:
        """Returns the price from the last trade if available
        or self.current_price if no trade found."""
        last_trade = self.last_trade
        return self.current_price if last_trade is None else last_trade.price

    '''def simulate_event_1(self, row: Series) -> float:
        """Simulates as an individual sample / data frame row comes in."""
        # save current event
        self.current_row = row
        # read the new side to change to
        side = self.new_side
        # get the roi
        roi = self.simulate_roi(row)
        # TODO: how do I know that a real trade happened?
        wallet = self.wallet * roi
        wallet_fee = self.wallet_fee * roi * self.x_fee
        # set roi, wallet and wallet fee on the row
        row[self.roi_col] = roi
        row[self.wallet_col] = wallet
        row[self.wallet_fee_col] = wallet_fee
        # store state of event
        self.side = side if self.is_side(side) else self.side
        self.wallet = wallet
        self.wallet_fee = wallet_fee
        self.last_row = row
        return row

    def is_side(self, side) -> bool:
        """Returns True if the given argument is a type of Side."""
        return side in Side.__args__

    @property
    def new_side(self) -> Side:
        """Return the side from current row, which is the new side to change to."""
        try:
            side = self.current_row[self.side_col]
        except KeyError:
            side = None
        return side if self.is_side(side) else None
'''


class MarketDFTradeSimulator(DFTradeSimulator):
    def should_trade(self):
        # trade should be done if current side does not match the signal
        # and if signal is not None
        return self.side != self.signal and self.signal is not None


class StopLimitDFTradeSimulator(MarketDFTradeSimulator):
    def __init__(self, df: DataFrame, treshold: float, **kwargs) -> None:
        self.treshold = treshold
        super().__init__(df, **kwargs)

    def should_trade(self):
        # get prices
        roi = self.calc_roi()
        change = 1 - roi

        self.current_row["stop-limit"] = "[ ]"

        # cause a stop limit if above the treshold
        if change > self.treshold and self.signal is None:
            self.current_row[self.side_col] = SELL if self.side == BUY else BUY
            self.current_row["stop-limit"] = "[x]"
            return True

        return super().should_trade()


if __name__ == "__main__":
    TEST_CSV = "./tests/datasets/test-small-results.csv"

    df = pd.read_csv(TEST_CSV, index_col="time", parse_dates=["time"])

    trade_sim = StopLimitDFTradeSimulator(df, treshold=0.1)
    trade_sim.simulate()
    print(trade_sim.df)
