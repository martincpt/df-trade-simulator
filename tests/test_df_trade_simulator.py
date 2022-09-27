import prepare_test_env
import unittest

from unittest.mock import patch

import numpy as np
import pandas as pd

from pandas import DataFrame, Series

from df_trade_simulator import DFTrade, DFTradeSimulator
from df_trade_simulator import Side
from df_trade_simulator import BUY, SELL, HODL


class DFTradeSimulator_TestCase(unittest.TestCase):
    df: DataFrame
    trade_simulator: DFTradeSimulator
    buy_sides: Series
    sell_sides: Series

    @patch.multiple(DFTradeSimulator, __abstractmethods__=set())
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

        self.assertTrue(all([x in trade_sim.df for x in trade_sim.columns]))

    def test_add_sides(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.add_sides(self.buy_sides, self.sell_sides)

        count = trade_sim.df[trade_sim.side_col].value_counts()
        buys = getattr(count, BUY, None)
        sells = getattr(count, SELL, None)

        self.assertEqual(buys, self.buy_sides_len)
        self.assertEqual(sells, self.sell_sides_len)

    def test_simulate_event_pre(self) -> None:
        row = self.df.iloc[0]
        trade_sim = self.trade_simulator
        trade_sim.simulate_event_pre(row)

        self.assertTrue(trade_sim.current_row.equals(row))

    def test_simulate_event_post(self) -> None:
        row = self.df.iloc[0]
        trade_sim = self.trade_simulator
        trade_sim.simulate_event_post(row)

        self.assertTrue(trade_sim.last_row.equals(row))

    def test_nan_safe(self) -> None:
        value = "safe"
        value_safe = self.trade_simulator.nan_safe(value)

        nan = float("NaN")
        none = self.trade_simulator.nan_safe(nan)

        self.assertEqual(value, value_safe)
        self.assertEqual(none, None)

    def test_extend_row_with_trade(self) -> None:
        row = self.df.iloc[0]
        trade = DFTrade(row, price=1, side=BUY, roi=1, wallet=2, wallet_fee=1.990)
        trade_sim = self.trade_simulator

        self.trade_simulator.extend_row_with_trade(row, trade)

        self.assertAlmostEqual(row[trade_sim.roi_col], 1)
        self.assertAlmostEqual(row[trade_sim.wallet_col], 2)
        self.assertAlmostEqual(row[trade_sim.wallet_fee_col], 1.990)

    @patch.multiple(DFTradeSimulator, __abstractmethods__=set())
    def test_simulate_event(self) -> None:
        csv = prepare_test_env.TEST_SMALL_RESULTS_CSV
        df = prepare_test_env.READ_TEST_CSV(csv)
        # prepare mock
        init_roi = 2
        roi = init_roi
        row = None
        should_trade_mock = lambda: roi % 2 != 0
        do_trade_mock = lambda: DFTrade(
            row=row,
            price=trade_sim.current_price,
            roi=roi,
            side=trade_sim.signal,
            wallet=trade_sim.wallet * roi,
            wallet_fee=trade_sim.wallet_fee * roi * trade_sim.x_fee,
        )
        # create instance and patch
        trade_sim = DFTradeSimulator(df)
        trade_sim.should_trade = should_trade_mock
        trade_sim.do_trade = do_trade_mock
        # iter through
        for index, row in trade_sim.df.iterrows():
            trade_sim.simulate_event(row)
            if trade_sim.should_trade():
                trade = trade_sim.last_trade
                wallet = np.prod([x.roi for x in trade_sim.trades])
                wallet_fee = wallet * trade_sim.x_fee ** len(trade_sim.trades)

                self.assertAlmostEqual(trade.roi, roi)
                self.assertAlmostEqual(trade.wallet, wallet)
                self.assertAlmostEqual(trade.wallet_fee, wallet_fee)
            # update roi
            roi += 1

    """def test_is_side(self) -> None:
        trade_sim = self.trade_simulator
        valids = []
        for arg in Side.__args__:
            valids.append(trade_sim.is_side(arg))

        invalid_val = "-".join(Side.__args__)
        invalid = trade_sim.is_side(invalid_val)

        self.assertTrue(all(valids))
        self.assertFalse(invalid)

    def test_new_side(self) -> None:
        trade_sim = self.trade_simulator
        df_mock = pd.DataFrame({trade_sim.side_col: Side.__args__})
        valids_without_nones = []

        for index, row_mock in df_mock.iterrows():
            trade_sim.current_row = row_mock
            valids_without_nones.append(trade_sim.new_side)

        trade_sim.current_row[trade_sim.side_col] = float("NaN")
        not_defined = trade_sim.new_side

        del trade_sim.current_row[trade_sim.side_col]
        col_missing = trade_sim.new_side

        self.assertTrue(None not in valids_without_nones)
        self.assertTrue(not_defined is None)
        self.assertTrue(col_missing is None)

    @patch.multiple(DFTradeSimulator, __abstractmethods__=set())
    def test_simulate_event_1(self) -> None:
        csv = prepare_test_env.TEST_SMALL_RESULTS_CSV
        df = prepare_test_env.READ_TEST_CSV(csv)
        # prepare mock
        init_roi = 2
        roi = init_roi
        simulate_roi_mock = lambda x: roi
        # create instance and patch
        trade_sim = DFTradeSimulator(df)
        trade_sim.simulate_roi = simulate_roi_mock
        # iter through
        for index, row in trade_sim.df.iterrows():
            updated = trade_sim.simulate_event(row)
            new_side = trade_sim.new_side
            side = trade_sim.side if new_side is None else new_side
            roi_list = np.arange(init_roi, roi + 1, 1)
            wallet = np.prod(roi_list)
            wallet_fee = wallet * (trade_sim.x_fee ** len(roi_list))
            # asserts
            self.assertEqual(id(row), id(trade_sim.current_row))
            self.assertEqual(trade_sim.side, side)
            self.assertEqual(updated[trade_sim.roi_col], roi)
            self.assertAlmostEqual(trade_sim.wallet, wallet)
            self.assertAlmostEqual(trade_sim.wallet_fee, wallet_fee)
            # update roi
            roi += 1

        # print(trade_sim.__dict__)
        # df = trade_sim.df
        # row = df.loc[~pd.isna(df[trade_sim.side_col])].iloc[0]
        # print(row)
        # print(row["side"][1])
        # trade_sim.simulate_event(row)
"""


if __name__ == "__main__":
    unittest.main()
