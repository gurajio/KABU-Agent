import unittest

import numpy as np
import pandas as pd

from functions import get_valuation_data
from ml import FeatureConfig, build_features
from simulation import simulate
from train import FEATURE_PARAMS


class SimulationTests(unittest.TestCase):
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
        self.assertTrue(features.iloc[13:19].return_mean.isna().all())

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
        )
        self.assertEqual(skipped, 1)
        self.assertEqual(trades, [])
        self.assertEqual(portfolio["cash"], 1_000_000)
        self.assertEqual(equity[-1]["total_assets"], 1_000_000)


if __name__ == "__main__":
    unittest.main()
