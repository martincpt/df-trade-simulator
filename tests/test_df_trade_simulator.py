import prepare_test_env
import unittest

from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd

from pandas import DataFrame, Series

from df_trade_simulator import (
    DFTrade,
    DFTradeSimulator,
    MarketDFTradeSimulator,
    StopLimitDFTradeSimulator,
)
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
        self.test_trade_row["price"] = self.test_trade_price
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
        # test seq
        self.test_trades_for_calc_roi = (
            # (signal, price, expected roi)
            ("sell", 200, 1),
            ("buy", 100, 200 / 100),
            ("sell", 75, 75 / 100),
        )

    def get_modded_row_and_trade(self, row: Series, **kwargs) -> tuple[Series, DFTrade]:
        row = row.copy()
        row["signal"] = None
        # mod row by kwargs
        for key, val in kwargs.items():
            if key in row:
                row[key] = kwargs[key]
        # init roi memory for this session
        roi_memory = "get_modded_row_and_trade_roi_memory"
        if not hasattr(self, roi_memory):
            setattr(self, roi_memory, [])
        # read rois
        rois = getattr(self, roi_memory)
        # check for trade kwargs
        trade_kwargs = {
            "roi": 1,
            "wallet": np.prod(rois),
            "wallet_fee": np.prod(rois) * self.trade_simulator.x_fee ** len(rois),
        }
        # update trade kwargs
        for key, val in trade_kwargs.items():
            trade_kwargs[key] = kwargs.get(key, val)
        # update rois
        rois.append(trade_kwargs["roi"])
        # create the corresponding trade object
        trade = DFTrade(row=row, price=row["price"], side=row["signal"], **trade_kwargs)

        return row, trade

    def test_set_df(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.set_df(self.df)

        # remove custom columns
        remove_columns = [x for x in trade_sim.columns if x not in self.df]
        compare_df = trade_sim.df.drop(columns=remove_columns)

        self.assertTrue(compare_df.equals(self.df))

    def test_add_signals(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.add_signals(self.buy_signals, self.sell_signals)

        count = trade_sim.df[trade_sim.signal_col].value_counts()
        buys = getattr(count, BUY, None)
        sells = getattr(count, SELL, None)

        self.assertEqual(buys, self.buy_signals_len)
        self.assertEqual(sells, self.sell_signals_len)

    def test_simulate_event_pre(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.simulate_event_pre(self.row)

        self.assertTrue(trade_sim.current_row.equals(self.row))
        self.assertTrue(trade_sim.last_row is None)

    def test_simulate_event_post(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.simulate_event_post(self.row)

        self.assertTrue(trade_sim.last_row.equals(self.row))
        self.assertTrue(trade_sim.current_row is None)

    def test_current_and_last_trade_price_defaults(self) -> None:
        trade_sim = self.trade_simulator

        # scenario 1: they all have initial value, which is None
        self.assertTrue(trade_sim.current_row is None)
        self.assertTrue(trade_sim.current_price is None)
        self.assertTrue(trade_sim.last_trade is None)
        self.assertTrue(trade_sim.last_trade_price is None)

    def test_current_and_last_trade_price_after_simulate_event(self) -> None:
        trade_sim = self.trade_simulator

        # scenario 2: simulate_event has been runned at least once
        trade_sim.simulate_event(self.row)

        # now current_row has been assigned, and current_price
        # should have the same value as the input row
        self.assertTrue(trade_sim.current_row.equals(self.row))
        self.assertAlmostEqual(trade_sim.current_price, self.row["price"])

        # also, if trade not happened but current_row is available
        # last_trade_price should return current_price as a fallback
        self.assertTrue(trade_sim.last_trade is None)
        self.assertAlmostEqual(trade_sim.last_trade_price, trade_sim.current_price)

    def test_last_trade_price_after_a_trade(self) -> None:
        trade_sim = self.trade_simulator

        # scenario 3: a trade is available
        trade_sim.add_trade(self.test_trade)

        self.assertTrue(trade_sim.last_trade == self.test_trade)
        self.assertAlmostEqual(self.test_trade_price, trade_sim.last_trade_price)

    def test_wallet_defaults(self) -> None:
        trade_sim = self.trade_simulator

        self.assertTrue(trade_sim.last_trade is None)
        self.assertTrue(isinstance(trade_sim.wallet, int | float))
        self.assertTrue(isinstance(trade_sim.wallet_fee, int | float))

    def test_wallet_after_a_trade(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.add_trade(self.test_trade)

        self.assertTrue(trade_sim.last_trade is not None)
        self.assertAlmostEqual(trade_sim.wallet, self.test_trade.wallet)
        self.assertAlmostEqual(trade_sim.wallet_fee, self.test_trade.wallet_fee)

    def test_side_default(self) -> None:
        trade_sim = self.trade_simulator

        self.assertTrue(trade_sim.side is None)

    def test_side_after_a_trade(self) -> None:
        trade_sim = self.trade_simulator
        trade_sim.add_trade(self.test_trade)

        self.assertEqual(trade_sim.side, self.test_trade.side)

    def test_signal_default(self) -> None:
        trade_sim = self.trade_simulator

        self.assertTrue(trade_sim.signal is None)

    def test_signal_after_simulate_event(self) -> None:
        self.row["signal"] = "sell"
        trade_sim = self.trade_simulator
        trade_sim.simulate_event(self.row)

        self.assertEqual(trade_sim.signal, self.row["signal"])

    def test_calc_roi_default(self) -> None:
        trade_sim = self.trade_simulator
        roi = trade_sim.calc_roi()

        self.assertAlmostEqual(roi, 1)

    def test_calc_roi(self) -> None:
        trade_sim = self.trade_simulator
        for signal, price, expected_roi in self.test_trades_for_calc_roi:
            row, trade = self.get_modded_row_and_trade(
                self.row, signal=signal, price=price
            )
            trade_sim.simulate_event(row)
            roi = trade_sim.calc_roi()
            trade_sim.add_trade(trade)

            self.assertAlmostEqual(roi, expected_roi)

    @patch("df_trade_simulator.DFTradeSimulator.wallet", new_callable=PropertyMock)
    def test_calc_wallet(self, wallet_mock) -> None:
        wallet_mock.return_value = 2
        trade_sim = self.trade_simulator
        wallet = trade_sim.calc_wallet(2, fee=False)
        wallet_fee = trade_sim.calc_wallet(2, fee=True)

        self.assertAlmostEqual(wallet, 2 * 2)
        self.assertAlmostEqual(wallet_fee, 2 * 2 * trade_sim.x_fee)

    def test_nan_safe(self) -> None:
        safe = "safe"
        safe_value = self.trade_simulator.nan_safe(safe)

        not_safe = float("NaN")
        none = self.trade_simulator.nan_safe(not_safe)

        self.assertEqual(safe, safe_value)
        self.assertEqual(none, None)

    def test_extend_row_with_trade(self) -> None:
        row = self.row.copy()
        trade = DFTrade(row, price=1, side=BUY, roi=1, wallet=2, wallet_fee=1.990)
        trade_sim = self.trade_simulator

        self.trade_simulator.extend_row_with_trade(row, trade)

        self.assertAlmostEqual(row[trade_sim.roi_col], 1)
        self.assertAlmostEqual(row[trade_sim.wallet_col], 2)
        self.assertAlmostEqual(row[trade_sim.wallet_fee_col], 1.990)

    def test_do_trade_default(self) -> None:
        trade = self.trade_simulator.do_trade()

        self.assertTrue(isinstance(trade, DFTrade))
        self.assertTrue(trade.row is None)
        self.assertTrue(trade.price is None)
        self.assertTrue(trade.side is None)

    def test_do_trade_after_simulate_event(self) -> None:
        self.row["signal"] = "sell"
        self.trade_simulator.simulate_event(self.row)
        trade = self.trade_simulator.do_trade()

        self.assertTrue(isinstance(trade, DFTrade))
        self.assertTrue(trade.row.equals(self.row))

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

    @patch.multiple(DFTradeSimulator, __abstractmethods__=set())
    def test_df_trades(self) -> None:
        csv = prepare_test_env.TEST_RESULTS_CSV
        df = prepare_test_env.READ_TEST_CSV(csv)
        trade_sim = DFTradeSimulator(df)
        trade_sim.add_signals(
            buy=(df["price"] < df["price"].mean()),
            sell=(df["price"] > df["price"].mean()),
        )
        trades = trade_sim.df_trades
        # there should be no NaNs
        self.assertTrue(not trades[trade_sim.signal_col].isnull().any())
        # for further test, make sure it has at least two rows
        self.assertTrue(
            len(trades) > 2,
            "Please make sure df_trades returns with at least two rows.",
        )
        # do the opposite as df_trades: remove all non-consecutive items
        keep_cons_only = trades.loc[
            trades[trade_sim.signal_col] == trades[trade_sim.signal_col].shift()
        ].copy()
        # and that should be an empty frame
        # because that means results were non-consecutives
        self.assertTrue(len(keep_cons_only) == 0)


class MarketDFTradeSimulator_TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.csv = prepare_test_env.TEST_SMALL_RESULTS_CSV
        self.df = prepare_test_env.READ_TEST_CSV(self.csv)

    def test_market_df_trade_simulator(self) -> None:
        # NOTE: just for coverage right now
        # TODO: make more proper tests
        trade_sim = MarketDFTradeSimulator(self.df)
        trade_sim.simulate()

        self.assertAlmostEqual(trade_sim.wallet, 0.666667, 6)


class StopLimitDFTradeSimulator_TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.csv = prepare_test_env.TEST_SMALL_RESULTS_CSV
        self.df = prepare_test_env.READ_TEST_CSV(self.csv)

    def test_stop_limit_df_trade_simulator(self) -> None:
        trade_sim = StopLimitDFTradeSimulator(self.df, treshold=0.1)
        trade_sim.simulate()

        self.assertAlmostEqual(trade_sim.wallet, 1.851852, 6)


if __name__ == "__main__":
    unittest.main()
