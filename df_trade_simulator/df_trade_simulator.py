import pandas as pd

from pandas import DataFrame, Series

from abc import ABC, abstractmethod

BUY, SELL, HODL = "buy", "sell", "hodl"

PRICE_COL = "price"
WALLET_COL = "wallet"
WALLET_FEE_COL = "wallet_fee"
ROI_COL = "roi"
SIDE_COL = "side"


class DFTradeSimulator(ABC):
    df: DataFrame

    def __init__(self, df: DataFrame, **kwargs) -> None:
        # read kwargs
        self.fee = kwargs.get("fee", 0.1)
        self.price_col = kwargs.get("price_col", PRICE_COL)
        self.wallet_col = kwargs.get("wallet_col", WALLET_COL)
        self.wallet_fee_col = kwargs.get("wallet_fee_col", WALLET_FEE_COL)
        self.roi_col = kwargs.get("roi_col", ROI_COL)
        self.side_col = kwargs.get("side_col", SIDE_COL)
        # set data frame
        self.set_df(df)

    def set_df(self, df: DataFrame):
        """Sets the current data frame."""
        # make a copy
        self.df = df.copy()
        # add side col if not exists
        if self.side_col not in self.df:
            self.df[self.side_col] = float("nan")

    def add_sides(self, buy: Series, sell: Series) -> None:
        """Adds sides by Series."""
        self.df.loc[buy, self.side_col] = BUY
        self.df.loc[sell, self.side_col] = SELL

    def get_df_by_side(self, side: str) -> DataFrame:
        """Returns a data frame where only the given side value occours."""
        return self.df.loc[self.df[self.side_col] == side]

    def get_roi(self, row: Series) -> float:
        return float("NaN")

    def simulate_event(self, row: Series) -> float:
        # print(row, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        row["roi"] = float("NaN")
        return row

    def simulate(self) -> None:
        """Simulates the state of the wallet through the data frame."""
        self.df = self.df.apply(self.simulate_event, axis=1)

        print(self.df)

    @property
    def buys(self) -> DataFrame:
        return self.get_df_by_side(BUY)

    @property
    def sells(self) -> DataFrame:
        return self.get_df_by_side(SELL)


class MarketDFTradeSimulator(DFTradeSimulator):
    """"""


if __name__ == "__main__":
    TEST_CSV = "./tests/datasets/small-test-results.csv"

    df = pd.read_csv(TEST_CSV, index_col="time", parse_dates=["time"])
    # print(df)
    trade_sim = MarketDFTradeSimulator(df)
    trade_sim.simulate()
