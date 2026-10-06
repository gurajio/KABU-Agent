import unittest

import numpy as np
import pandas as pd

from functions import get_valuation_data, order_reason
from ml import FeatureConfig, build_features
from simulation import decision_prices, prediction_accuracy, simulate
from train import FEATURE_PARAMS


class SimulationTests(unittest.TestCase):
    def test_prediction_accuracy(self):
        index = pd.date_range("2026-09-08 09:00", periods=6, freq="5min", tz="Asia/Tokyo")
        close = np.array([100., 100., 100., 110., 90., 100.])
        frame = pd.DataFrame(dict(Open=close, High=close + 1, Low=close - 1,
                                  Close=close, Volume=100), index=index)
        times = index + pd.Timedelta(minutes=5)
        probability = pd.Series([0.5, 0.4, 0.6, 0.8, 0.8, np.nan], index=times)
        metrics = prediction_accuracy({"A": frame}, {"A": probability}, FeatureConfig(**FEATURE_PARAMS),
                                      horizon=2, start=times[0], end=times[-1])
        self.assertEqual(metrics["evaluated_count"], 3)
        self.assertEqual(metrics["correct_count"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        shortened = prediction_accuracy({"A": frame}, {"A": probability}, FeatureConfig(**FEATURE_PARAMS),
                                        horizon=2, start=times[0], end=index[3])
        self.assertEqual(shortened["evaluated_count"], 0)
        self.assertIsNone(shortened["accuracy"])

    def setUp(self):
        index = pd.date_range("2026-09-08 09:00", periods=24, freq="5min", tz="Asia/Tokyo")
        close = 100 + np.arange(24) + np.sin(np.arange(24))
        self.frame = pd.DataFrame(dict(Open=close, High=close + 2, Low=close - 2, Close=close + 0.5, Volume=100 + np.arange(24) ** 2), index=index)

    def test_past_only(self):
        config = FeatureConfig(**FEATURE_PARAMS)
        market = self.frame.copy()
        cutoff = self.frame.index[17] + pd.Timedelta(minutes=5)
        before = build_features(self.frame, market, config, as_of=cutoff)
        future = self.frame.copy()
        future.loc[future.index > self.frame.index[17], ["Open", "High", "Low", "Close"]] *= 3
        after = build_features(future, future, config, as_of=cutoff)
        pd.testing.assert_frame_equal(before, after)

    def test_missing_breaks(self):
        frame = self.frame.copy()
        frame.iloc[12] = np.nan
        features = build_features(frame, self.frame, FeatureConfig(**FEATURE_PARAMS))
        self.assertTrue(features.iloc[12].isna().all())
        self.assertTrue(features.iloc[13:19].return_30m.isna().all())

    def test_valuation(self):
        frame = self.frame.copy()
        frame.loc[frame.index[-1], "Close"] = np.nan
        prices, stamp = get_valuation_data({"9432.T": frame}, {"positions": {"9432.T": {"quantity": 100}}}, frame.index[-1])
        self.assertEqual(prices["9432.T"]["latest_price"], frame.Close.iloc[-2])
        self.assertEqual(prices["9432.T"]["price_datetime"], frame.index[-2])
        self.assertEqual(stamp, frame.index[-1] + pd.Timedelta(minutes=5))

    def test_missing_open(self):
        frame = self.frame.copy()
        frame.loc[frame.index[0], "Open"] = np.nan
        portfolio, trades, equity, skipped = simulate(
            {"9432.T": frame}, {"9432.T": pd.Series(dtype=float)}, strategy="rf", start=frame.index[0],
            end=frame.index[-1] + pd.Timedelta(minutes=5), buy_threshold=0.55, sell_threshold=0.45,
            initial_cash=1_000_000,
        )
        self.assertEqual(skipped, 1)
        self.assertEqual(trades, [])
        self.assertEqual(portfolio["cash"], 1_000_000)
        self.assertEqual(equity[-1]["total_assets"], 1_000_000)

    def test_cash_reserve(self):
        index = self.frame.index[:3]
        frame = pd.DataFrame(dict(Open=900., High=900., Low=900., Close=900., Volume=100), index=index)
        prices = {str(i): frame.copy() for i in range(10)}
        probabilities = {symbol: pd.Series(0.8, index=index) for symbol in prices}
        counts = {}
        portfolio, trades, _, _ = simulate(
            prices, probabilities, strategy="rf", start=index[0],
            end=index[-1] + pd.Timedelta(minutes=5), buy_threshold=0.55, sell_threshold=0.45,
            initial_cash=1_000_000, diagnostics=counts,
        )
        self.assertEqual(portfolio["cash"], 100_000)
        self.assertEqual(len(trades), 10)
        self.assertEqual(counts["reserve_limit"], 20)
        self.assertTrue(all(t["cash_ratio_after"] >= 0.1 for t in trades))
        self.assertTrue(all(t["position_ratio_after"] <= 0.1 for t in trades))

    def test_position_limit(self):
        portfolio = {"cash": 900_000, "positions": {"A": {"quantity": 500, "entry_price": 100}}}
        order = dict(symbol="A", quantity=100, price=200, cost=0, action="buy")
        self.assertEqual(order_reason(
            portfolio, order, set(), latest_prices={"A": 200},
            reserve_ratio=0.1, max_position_ratio=0.1,
        ), "position_limit")
        order["action"] = "sell"
        self.assertIsNone(order_reason(
            portfolio, order, set(), latest_prices={"A": 200},
            reserve_ratio=0.1, max_position_ratio=0.1,
        ))

    def test_limits_with_cost(self):
        portfolio = {"cash": 1_000_000, "positions": {}}
        order = dict(symbol="A", quantity=100, price=1000, cost=0, action="buy")
        self.assertIsNone(order_reason(portfolio, order, set(), reserve_ratio=0.1, max_position_ratio=0.1))
        order["cost"] = 1
        self.assertEqual(order_reason(portfolio, order, set(), max_position_ratio=0.1), "position_limit")
        order.update(price=9000, cost=1000)
        self.assertEqual(order_reason(portfolio, order, set(), reserve_ratio=0.1), "reserve_limit")

    def test_decision_prices(self):
        frame = self.frame.iloc[:3].copy()
        frame.loc[frame.index[1], "Open"] = np.nan
        marks = decision_prices({"A": frame}, frame.index)
        self.assertEqual(marks.at[frame.index[1], "A"], frame.Close.iloc[0])
        changed = frame.copy()
        changed.loc[changed.index >= frame.index[1], "Close"] *= 10
        after = decision_prices({"A": changed}, changed.index)
        self.assertEqual(after.at[frame.index[1], "A"], marks.at[frame.index[1], "A"])

    def test_invalid_limits(self):
        for options in ({"reserve_ratio": float("nan")}, {"max_position_ratio": 0}, {"initial_cash": -1}):
            with self.assertRaises(ValueError):
                simulate({}, {}, strategy="rf", start=self.frame.index[0], end=self.frame.index[-1],
                         buy_threshold=0.55, sell_threshold=0.45, **options)


if __name__ == "__main__":
    unittest.main()
