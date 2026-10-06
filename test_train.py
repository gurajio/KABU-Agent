import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml import FEATURE_COLUMNS, FeatureConfig, build_features, load_model
from ml.models import build_samples, load_model_map
import train
from train import FEATURE_PARAMS, PERIODS, TRAIN_END, compare_models, compare_symbols, fit_progress, main
from ml.models import fit_classifier


class TrainingTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(train, "tree_steps"), "コード設定優先機能は現行train.pyに含まれていません")
    def test_code_parameters_take_priority(self):
        params = dict(n_estimators=200, max_depth=3, min_samples_leaf=10, random_state=43)
        with tempfile.TemporaryDirectory() as directory:
            for per_symbol in (False, True):
                argv = ["train.py", "--max-depth", "9", "--min-samples-leaf", "1"]
                argv += (["--per-symbol", "--output", str(Path(directory) / "comparison")] if per_symbol
                         else ["--model", str(Path(directory) / "model.pkl")])
                with (patch("sys.argv", argv), patch("train.FOREST_PARAMS", params),
                      patch("train.load_data", return_value=({"A": None}, None)),
                      patch("train.data_hash", return_value="test-data"),
                      patch("train.compare_symbols") as compare,
                      patch("train.fit_model", return_value=SimpleNamespace()) as fit,
                      patch("train.save_model")):
                    main()
                    if per_symbol:
                        self.assertEqual(compare.call_args.args[-1], params)
                    else:
                        self.assertEqual(fit.call_args.kwargs["forest_params"], params)
        self.assertEqual(train.tree_steps(params), (10, 20, 50, 100, 200))
        self.assertEqual(train.tree_steps(dict(n_estimators=15)), (10, 15))
        self.assertEqual(train.tree_steps(dict(n_estimators=5)), (5,))

    def test_symbol_models_and_mapping(self):
        symbols = ["A", "B", "C"]
        index = pd.MultiIndex.from_product(
            [symbols, pd.date_range("2026-09-08 10:00", periods=12, freq="5min", tz="Asia/Tokyo")],
            names=["symbol", "decision_time"],
        )
        features = pd.DataFrame(np.random.default_rng(43).normal(size=(36, len(FEATURE_COLUMNS))),
                                index=index, columns=FEATURE_COLUMNS)
        target = pd.Series([0, 1] * 12 + [0] * 12, index=index)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "symbols"
            with (patch("train.build_samples", return_value=(features, target)) as samples,
                  patch("train.data_hash", return_value="test-data"), patch("train.WORKERS", 1)):
                compare_symbols(dict.fromkeys(symbols), None, FeatureConfig(**FEATURE_PARAMS), output,
                                "test-data", dict(n_estimators=3, max_depth=2, random_state=43))
            shared = load_model_map(output / "selection.json")
            individual = load_model_map(output / "individual_models.json")
            self.assertEqual(shared["A"].symbols, tuple(symbols))
            self.assertIs(shared["A"], shared["B"])
            self.assertEqual(individual["A"].symbols, ("A",))
            self.assertEqual(individual["C"].symbols, tuple(symbols))
            self.assertFalse((output / "C.pkl").exists())
            comparison = pd.read_csv(output / "symbol_comparisons.csv").set_index("symbol")
            self.assertEqual(comparison.loc["C", "status"], "skipped")
            self.assertEqual(comparison.loc["A", "validation_count"], 12)
            predictions = pd.read_csv(output / "A_predictions.csv")
            self.assertTrue({"target", "common_probability", "individual_probability"}.issubset(predictions.columns))
            settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["trained_symbols"], 2)
            overall = settings["overall_validation"]
            self.assertEqual(overall["evaluated_count"], 24)
            self.assertEqual(overall["evaluated_symbols"], 2)
            self.assertEqual(overall["individual_correct"], int(comparison.loc[["A", "B"], "individual_correct"].sum()))
            self.assertAlmostEqual(overall["individual_accuracy"], overall["individual_correct"] / 24)
            self.assertAlmostEqual(overall["common_accuracy"], overall["common_correct"] / 24)
            self.assertFalse(settings["test_used"])
            self.assertFalse(settings["model_adopted"])
            self.assertEqual(samples.call_args_list[0].kwargs["end"], TRAIN_END)
            self.assertEqual(samples.call_args_list[1].kwargs["end"], PERIODS["validation"][1])
            wrong = output / "wrong.json"
            wrong.write_text(json.dumps({"A": "B.pkl"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_map(wrong)
            individual["B"].horizon = 3
            from ml import save_model
            save_model(individual["B"], output / "B_other.pkl")
            wrong.write_text(json.dumps({"A": "A.pkl", "B": "B_other.pkl"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_map(wrong)

    def test_tree_progress(self):
        features = pd.DataFrame(np.random.default_rng(43).normal(size=(24, len(FEATURE_COLUMNS))),
                                columns=FEATURE_COLUMNS)
        target = pd.Series([0, 1] * 12)
        params = dict(n_estimators=6, max_depth=3, random_state=43)
        with patch("train.WORKERS", 1):
            classifier, history = fit_progress(features.iloc[:16], target.iloc[:16],
                                               features.iloc[16:], target.iloc[16:], params, steps=(2, 4, 6))
        direct = fit_classifier(features.iloc[:16], target.iloc[:16], forest_params=params, workers=1)
        np.testing.assert_array_equal(classifier.predict_proba(features), direct.predict_proba(features))
        self.assertEqual(len(classifier.estimators_), 6)
        self.assertEqual(history.n_estimators.tolist(), [2, 4, 6])
        self.assertTrue(history.validation_count.eq(8).all())
        np.testing.assert_allclose(history.validation_accuracy, history.validation_correct / 8)
        with self.assertRaises(ValueError):
            fit_progress(features, target, features, target, params, steps=(4, 2))

    def feature_data(self):
        index = pd.date_range("2026-09-08 09:00", periods=24, freq="5min", tz="Asia/Tokyo")
        close = 100 + np.arange(24) + np.sin(np.arange(24))
        prices = pd.DataFrame(dict(Open=close - 0.5, High=close + 2, Low=close - 2,
                                   Close=close, Volume=100 + np.arange(24) ** 2), index=index)
        market = pd.DataFrame({"Close": 1000 + np.arange(24) ** 2}, index=index)
        return prices, market

    def test_momentum_formulas_and_causality(self):
        prices, market = self.feature_data()
        config = FeatureConfig(**FEATURE_PARAMS)
        result = build_features(prices, market, config)
        self.assertEqual(tuple(result.columns), FEATURE_COLUMNS)
        self.assertEqual(len(result.columns), 12)
        row = result.iloc[15]
        self.assertAlmostEqual(row.signed_body, 0.5 / prices.Open.iloc[15])
        for minutes, periods in ((5, 1), (15, 3), (30, 6)):
            self.assertAlmostEqual(row[f"return_{minutes}m"], prices.Close.iloc[15] / prices.Close.iloc[15-periods] - 1)
        excess = prices.Close.iloc[15] / prices.Close.iloc[12] - market.Close.iloc[15] / market.Close.iloc[12]
        self.assertAlmostEqual(row.excess_return_15m, excess)
        cutoff = result.index[15]
        changed = prices.copy()
        changed.loc[changed.index >= cutoff, ["Open", "High", "Low", "Close"]] *= 10
        changed_market = market.copy()
        changed_market.loc[changed_market.index >= cutoff, "Close"] *= 10
        pd.testing.assert_frame_equal(result.loc[:cutoff], build_features(changed, changed_market, config, as_of=cutoff))

    def test_history_resets_and_missing_market(self):
        prices, market = self.feature_data()
        config = FeatureConfig(**FEATURE_PARAMS)
        missing = market.drop(market.index[12])
        result = build_features(prices, missing, config)
        self.assertTrue(result.excess_return_15m.iloc[12:16].isna().all())
        for shift in (pd.Timedelta(hours=3, minutes=30), pd.Timedelta(days=1), pd.Timedelta(minutes=5)):
            interrupted = prices.copy()
            interrupted.index = interrupted.index[:12].append(interrupted.index[12:] + shift)
            result = build_features(interrupted, market, config)
            self.assertTrue(result.return_30m.iloc[12:18].isna().all())
        flat = prices.copy()
        flat.iloc[15, flat.columns.get_indexer(["Open", "High", "Low", "Close"])] = 100
        self.assertTrue(pd.isna(build_features(flat, market, config).close_position.iloc[15]))

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
        features = pd.DataFrame(np.random.default_rng(43).normal(size=(20, len(FEATURE_COLUMNS))),
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
            bundle = load_model(output / "depth_5_leaf_1.pkl")
            self.assertEqual(bundle.feature_names, FEATURE_COLUMNS)
            self.assertEqual(samples.call_args_list[0].kwargs["end"], TRAIN_END)
            self.assertEqual(samples.call_args_list[1].kwargs["start"], PERIODS["validation"][0])
            self.assertEqual(samples.call_args_list[1].kwargs["end"], PERIODS["validation"][1])


if __name__ == "__main__":
    unittest.main()
