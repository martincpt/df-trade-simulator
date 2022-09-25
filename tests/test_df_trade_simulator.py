import prepare_test_env
import unittest

from unittest.mock import patch

import pandas as pd

from pandas import DataFrame, Series

from df_trade_simulator import DFTradeSimulator
from df_trade_simulator import BUY, SELL, HODL


@patch.multiple(DFTradeSimulator, __abstractmethods__=set())
class DFTradeSimulator_TestCase(unittest.TestCase):
    df: DataFrame
    trade_simulator: DFTradeSimulator
    buy_sides: Series
    sell_sides: Series

    def setUp(self) -> None:
        self.df = prepare_test_env.READ_TEST_CSV()
        self.trade_simulator = DFTradeSimulator(self.df)
        self.buy_sides = self.df["pred"] == 2
        self.sell_sides = self.df["pred"] == 0
        self.buy_sides_len = len(self.buy_sides[self.buy_sides == True])
        self.sell_sides_len = len(self.sell_sides[self.sell_sides == True])

    def test_set_df(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.set_df(self.df)

        self.assertTrue(trade_sim.side_col in trade_sim.df)

    def test_add_sides(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.add_sides(self.buy_sides, self.sell_sides)

        buys = trade_sim.buys
        sells = trade_sim.sells

        self.assertEqual(len(buys), self.buy_sides_len)
        self.assertEqual(len(sells), self.sell_sides_len)

    def test_get_df_by_side(self) -> None:
        trade_sim = self.trade_simulator
        df = trade_sim.get_df_by_side(BUY)
        is_df = isinstance(df, DataFrame)

        self.assertTrue(is_df)


if __name__ == "__main__":
    unittest.main()
