import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ml import FeatureConfig, fit_model, save_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MARKET_CSV = Path("data") / "^N225_5m.csv"
TRAIN_START = "2026-08-07 00:00:00+09:00"
TRAIN_END = "2026-09-07 15:30:00+09:00"
PERIODS = {
    "validation": ("2026-09-08 00:00:00+09:00", "2026-09-17 15:30:00+09:00"),
    "test": ("2026-09-18 00:00:00+09:00", "2026-10-02 15:30:00+09:00"),
}
HORIZON = 2
WORKERS = min(4, max(1, (os.cpu_count() or 1) - 1))
MODEL_PATH = ROOT / "models" / "rf_current.pkl"
FEATURE_PARAMS = {
    "mean_window": 6,
    "std_window": 6,
    "autocorr_window": 6,
    "adx_window": 5,
    "adx_warmup": 0,
    "volume_window": 6,
    "corr_window": 6,
    "timestamp_kind": "start",
    "sessions": (("09:00", "11:30"), ("12:30", "15:30")),
    "market_name": "日経平均",
}
FOREST_PARAMS = {"n_estimators": 100, "max_depth": 5, "random_state": 43}


def read_prices(path, *, end=TRAIN_END):
    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or frame.index.hasnans:
        raise ValueError(f"タイムゾーン付きの欠損のない日時が必要です: {path}")
    frame.index = frame.index.tz_convert("Asia/Tokyo")
    frame = frame.loc[
        (frame.index >= pd.Timestamp(TRAIN_START))
        & (frame.index + pd.Timedelta(minutes=5) <= pd.Timestamp(end))
    ]
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"データが空、または日時の重複・順序異常があります: {path}")
    if (frame.index != frame.index.floor("5min")).any():
        raise ValueError(f"5分足の開始時刻と一致しません: {path}")
    frame = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]].astype(float)
    prices = frame.drop(columns="Volume")
    invalid = (
        (prices <= 0).any().any() or (frame.Volume < 0).any()
        or np.isinf(frame.to_numpy()).any() or (frame.High < frame.Low).any()
        or (frame.High < frame[["Open", "Close"]].max(axis=1)).any()
        or (frame.Low > frame[["Open", "Close"]].min(axis=1)).any()
    )
    if invalid:
        raise ValueError(f"価格・出来高に不正な値があります: {path}")
    if not frame.notna().all(axis=1).any():
        raise ValueError(f"有効なOHLCVの足がありません: {path}")
    return frame


def load_data(*, end=TRAIN_END):
    symbols = pd.read_csv(ROOT / "tickers.csv")["symbol"]
    if symbols.empty or symbols.isna().any() or symbols.duplicated().any():
        raise ValueError("tickers.csvの銘柄に欠損・重複があります。")
    prices = {symbol: read_prices(DATA_DIR / f"{symbol}_5m.csv", end=end) for symbol in symbols}
    market = read_prices(ROOT / MARKET_CSV, end=end)
    for symbol, frame in [*prices.items(), ("^N225", market)]:
        missing = int(frame.isna().any(axis=1).sum())
        print(f"データ確認 {symbol}: {len(frame)}本、{frame.index[0]}〜{frame.index[-1]}、欠損行={missing}")
    print("欠損は補完しません。取得条件・休場日を含む網羅性はCSVだけでは確認できません。")
    return prices, market


def data_hash():
    symbols = pd.read_csv(ROOT / "tickers.csv")["symbol"].tolist()
    digest = hashlib.sha256()
    for path in [ROOT / "tickers.csv", *[DATA_DIR / f"{s}_5m.csv" for s in sorted(symbols)], ROOT / MARKET_CSV]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="保存済みCSVからRFモデルを学習します。")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    destination = (ROOT / args.model).resolve()
    config = FeatureConfig(**FEATURE_PARAMS)
    if destination.exists():
        raise FileExistsError(f"保存先が存在します。--modelで別名を指定してください: {destination}")
    fingerprint = data_hash()
    prices, market = load_data()
    print(f"学習期間: {TRAIN_START} 〜 {TRAIN_END}、銘柄数: {len(prices)}")
    model = fit_model(
        prices=prices, market=market, config=config,
        horizon=HORIZON, train_end=TRAIN_END,
        forest_params=FOREST_PARAMS, workers=WORKERS,
    )
    if data_hash() != fingerprint:
        raise ValueError("学習中にCSVが変更されたため、モデルを保存しません。")
    model.data_hash = fingerprint
    model.train_start = TRAIN_START
    save_model(model, destination)
    print(f"モデルを保存しました: {destination}")


if __name__ == "__main__":
    main()
