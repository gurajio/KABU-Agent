import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml import FEATURE_COLUMNS, FeatureConfig
from ml.models import build_samples
from train import FEATURE_PARAMS, PERIODS, TRAIN_END, compare_models


class TrainingTests(unittest.TestCase):
    def test_sample_boundary(self):
        index = pd.date_range("2026-09-07 09:00", periods=24, freq="5min", tz="Asia/Tokyo")
        close = 100 + np.arange(24) + np.sin(np.arange(24))
        frame = pd.DataFrame(dict(Open=close, High=close + 2, Low=close - 2,
                                  Close=close + 0.5, Volume=100 + np.arange(24) ** 2), index=index)
        cutoff = index[17] + pd.Timedelta(minutes=5)
        config = FeatureConfig(**FEATURE_PARAMS)
        before = build_samples({"A": frame}, frame, config, horizon=2, end=cutoff)
        future = frame.copy()
        future.loc[future.index >= cutoff, ["Open", "High", "Low", "Close"]] *= 10
        after = build_samples({"A": future}, future, config, horizon=2, end=cutoff)
        pd.testing.assert_frame_equal(before[0], after[0])
        pd.testing.assert_series_equal(before[1], after[1])
        self.assertLessEqual(before[0].index.get_level_values("decision_time").max(), index[15])

    def test_compare_no_adoption(self):
        index = pd.MultiIndex.from_product(
            [["A"], pd.date_range("2026-09-08 10:00", periods=20, freq="5min", tz="Asia/Tokyo")],
            names=["symbol", "decision_time"],
        )
        features = pd.DataFrame(np.random.default_rng(43).normal(size=(20, 10)),
                                index=index, columns=FEATURE_COLUMNS)
        target = pd.Series([0, 1] * 10, index=index)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current.pkl"
            current.write_bytes(b"existing-model")
            output = root / "comparison"
            with (patch("train.build_samples", return_value=(features, target)) as samples,
                  patch("train.data_hash", return_value="test-data"),
                  patch("train.MODEL_PATH", current), patch("train.WORKERS", 1),
                  patch("train.FOREST_PARAMS", {"n_estimators": 3, "random_state": 43}),
                  patch("train.MODEL_CANDIDATES", ((5, 1), (10, 2)))):
                compare_models({"A": None}, None, FeatureConfig(**FEATURE_PARAMS), output, "test-data")
            settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["status"], "complete")
            self.assertFalse(settings["model_adopted"])
            self.assertFalse(settings["test_used"])
            self.assertEqual(current.read_bytes(), b"existing-model")
            self.assertEqual(len(pd.read_csv(output / "comparisons.csv")), 2)
            self.assertEqual(samples.call_args_list[0].kwargs["end"], TRAIN_END)
            self.assertEqual(samples.call_args_list[1].kwargs["start"], PERIODS["validation"][0])
            self.assertEqual(samples.call_args_list[1].kwargs["end"], PERIODS["validation"][1])


if __name__ == "__main__":
    unittest.main()
