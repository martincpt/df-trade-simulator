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
    buy_signals: Series
    sell_signals: Series
    row: Series

    @patch.multiple(DFTradeSimulator, __abstractmethods__=set())
    def setUp(self) -> None:
        self.df = prepare_test_env.READ_TEST_CSV()
        self.trade_simulator = DFTradeSimulator(self.df)
        self.buy_signals = self.df["pred"] == 2
        self.sell_signals = self.df["pred"] == 0
        self.buy_signals_len = len(self.buy_signals[self.buy_signals == True])
        self.sell_signals_len = len(self.sell_signals[self.sell_signals == True])
        self.row = self.df.iloc[0]
        # test trade
        self.test_trade_price = 314.59
        self.test_trade_row = self.row.copy()
        self.test_trade_price["price"] = self.test_trade_price
        self.test_trade_side = "sell"
        self.test_trade_roi = 1.2
        self.test_trade_wallet = self.test_trade_roi
        self.test_trade_wallet_fee = self.test_trade_wallet * self.trade_simulator.x_fee
        self.test_trade = DFTrade(
            row=self.row,
            price=self.test_trade_price,
            side=self.test_trade_side,
            roi=1.2,
            wallet=self.test_trade_wallet,
            wallet_fee=self.test_trade_wallet_fee,
        )

    def test_set_df(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.set_df(self.df)

        self.assertTrue(all([x in trade_sim.df for x in trade_sim.columns]))

    def test_add_signals(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.add_signals(self.buy_signals, self.sell_signals)

        count = trade_sim.df[trade_sim.signal_col].value_counts()
        buys = getattr(count, BUY, None)
        sells = getattr(count, SELL, None)

        self.assertEqual(buys, self.buy_signals_len)
        self.assertEqual(sells, self.sell_signals_len)

    def test_simulate_event_pre(self) -> None:
        row = self.row.copy()
        trade_sim = self.trade_simulator
        trade_sim.simulate_event_pre(row)

        self.assertTrue(trade_sim.current_row.equals(row))

    def test_simulate_event_post(self) -> None:
        row = self.row.copy()
        trade_sim = self.trade_simulator
        trade_sim.simulate_event_post(row)

        self.assertTrue(trade_sim.last_row.equals(row))

    def test_current_and_last_trade_price(self) -> None:
        row = self.row.copy()
        trade_sim = self.trade_simulator
        trade_sim.should_trade = lambda: False

        # scenario 1: they both have initial value, which is None
        current_price_is_none = trade_sim.current_price
        self.assertTrue(trade_sim.current_row is None)
        self.assertTrue(current_price_is_none is None)

        last_price_is_none = trade_sim.last_trade_price
        self.assertTrue(trade_sim.last_row is None)
        self.assertTrue(last_price_is_none is None)

        # scenario 2: simulate_event_pre has been called
        # and current_row has been assigned,
        # so current_price will have the same value as in the row
        trade_sim.simulate_event(row)

        current_price_is_same = trade_sim.current_price
        self.assertTrue(trade_sim.current_row is not None)
        self.assertAlmostEqual(current_price_is_same, row["price"])

        # if trade not happened but current_row is available
        # last_trade_price should return current_price as a fallback
        last_price_is_same_as_current = trade_sim.last_trade_price
        self.assertTrue(trade_sim.last_trade is None)
        self.assertAlmostEqual(last_price_is_same_as_current, current_price_is_same)

        # scenario 3: a trade is available
        trade_sim.trades.append(self.test_trade)

        self.assertTrue(trade_sim.last_trade is not None)
        self.assertAlmostEqual(self.test_trade_price, trade_sim.last_trade_price)

    def test_calc_roi(self) -> None:
        return
        trade_sim = self.trade_simulator
        row = self.row.copy()
        print(">>>>>>>>>>>>")
        print(trade_sim.last_price)
        print(trade_sim.current_price)
        print(trade_sim.last_price)
        print(row)

    def test_nan_safe(self) -> None:
        value = "safe"
        value_safe = self.trade_simulator.nan_safe(value)

        nan = float("NaN")
        none = self.trade_simulator.nan_safe(nan)

        self.assertEqual(value, value_safe)
        self.assertEqual(none, None)

    def test_extend_row_with_trade(self) -> None:
        row = self.row.copy()
        trade = DFTrade(row, price=1, side=BUY, roi=1, wallet=2, wallet_fee=1.990)
        trade_sim = self.trade_simulator

        self.trade_simulator.extend_row_with_trade(row, trade)

        self.assertAlmostEqual(row[trade_sim.roi_col], 1)
        self.assertAlmostEqual(row[trade_sim.wallet_col], 2)
        self.assertAlmostEqual(row[trade_sim.wallet_fee_col], 1.990)

    def test_do_trade(self) -> None:
        pass

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


if __name__ == "__main__":
    unittest.main()
