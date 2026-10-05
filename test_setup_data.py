import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

try:
    import yfinance
except ModuleNotFoundError:
    yfinance = types.ModuleType("yfinance")
    yfinance.__version__ = "test-only"
    yfinance.config = types.SimpleNamespace(debug=types.SimpleNamespace(hide_exceptions=True))
    yfinance.Ticker = None
    exceptions = types.ModuleType("yfinance.exceptions")
    exceptions.YFRateLimitError = type("YFRateLimitError", (Exception,), {})
    sys.modules["yfinance"] = yfinance
    sys.modules["yfinance.exceptions"] = exceptions

import setup_data as setup


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2026-08-07", tz="Asia/Tokyo")
        self.end = pd.Timestamp("2026-10-03", tz="Asia/Tokyo")
        self.frame = pd.DataFrame(
            {"Open": [100.0], "High": [102.0], "Low": [99.0], "Close": [101.0], "Volume": [10.0]},
            index=pd.DatetimeIndex(["2026-08-07 09:00"], tz="Asia/Tokyo"),
        )

    def test_period(self):
        now = pd.Timestamp("2026-10-05 14:00", tz="Asia/Tokyo")
        self.assertEqual(setup.resolve_period("earliest", "2026-10-02", now), (self.start, self.end))
        with self.assertRaises(ValueError):
            setup.resolve_period("2026-08-05", "2026-10-02", now)

    def test_resume(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(setup, "fetch_data", return_value=self.frame) as fetch:
            output = Path(folder)
            self.assertEqual(setup.run_setup(["9432.T"], self.start, self.end, output, 2, 2, 60), 0)
            self.assertEqual(setup.run_setup(["9432.T"], self.start, self.end, output, 2, 2, 60), 0)
            self.assertEqual(fetch.call_count, 1)
            path = output / "9432.T_5m.csv"
            original = path.read_bytes()
            self.assertEqual(setup.run_setup(["9432.T"], self.start + pd.Timedelta(days=1), self.end, output, 2, 2, 60), 1)
            self.assertEqual(path.read_bytes(), original)

    def test_failure_continues(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(setup, "fetch_data", side_effect=[pd.DataFrame(), self.frame]), patch.object(setup.time, "sleep"):
            output = Path(folder)
            self.assertEqual(setup.run_setup(["9432.T", "^N225"], self.start, self.end, output, 2, 2, 60), 1)
            self.assertFalse((output / "9432.T_5m.csv").exists())
            self.assertTrue((output / "^N225_5m.csv").exists())

    def test_rate_stops(self):
        error = setup.YFRateLimitError()
        with tempfile.TemporaryDirectory() as folder, patch.object(setup, "fetch_data", side_effect=error) as fetch:
            output = Path(folder)
            self.assertEqual(setup.run_setup(["9432.T", "^N225"], self.start, self.end, output, 2, 2, 60), 1)
            self.assertEqual(fetch.call_count, 1)
            report = json.loads(next(output.glob("setup_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(report["results"][1]["status"], "not_attempted")

    def test_retry_limit(self):
        ticker = types.SimpleNamespace(history=unittest.mock.Mock(side_effect=setup.YFRateLimitError()))
        with patch.object(setup.yf, "Ticker", return_value=ticker), patch.object(setup.time, "sleep") as sleep:
            with self.assertRaises(setup.YFRateLimitError):
                setup.fetch_data("9432.T", self.start, self.end, 2, 60)
            self.assertEqual(ticker.history.call_count, 3)
            self.assertEqual(sleep.call_args_list, [unittest.mock.call(60), unittest.mock.call(120)])


if __name__ == "__main__":
    unittest.main()
