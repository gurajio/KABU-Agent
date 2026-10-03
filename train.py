from pathlib import Path

import pandas as pd

from ml import FeatureConfig, fit_model, save_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MARKET_CSV = Path("data") / "^N225_5m.csv"
TRAIN_START = "2026-08-05 00:00:00+09:00"
TRAIN_END = "2026-09-05 15:30:00+09:00"
HORIZON = 2
WORKERS = 4
MODEL_PATH = ROOT / "models" / "rf_v1.pkl"
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


def read_prices(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    return frame.loc[
        (frame.index >= pd.Timestamp(TRAIN_START))
        & (frame.index <= pd.Timestamp(TRAIN_END))
    ]


def main():
    if not MARKET_CSV or not FEATURE_PARAMS["market_name"]:
        raise ValueError("市場指数CSVの相対パスをMARKET_CSVに、指数名をmarket_nameに設定してください。")
    config = FeatureConfig(**FEATURE_PARAMS)
    if MODEL_PATH.exists():
        raise FileExistsError(f"保存先が存在します。MODEL_PATHを変更してください: {MODEL_PATH}")
    market_path = (ROOT / MARKET_CSV).resolve()
    market = read_prices(market_path)
    prices = {
        path.name.removesuffix("_5m.csv"): read_prices(path)
        for path in sorted(DATA_DIR.glob("*.T_5m.csv"))
        if path.resolve() != market_path
    }
    if not prices:
        raise ValueError("dataフォルダに銘柄の5分足CSVがありません。")
    print(f"読み込んだ銘柄数: {len(prices)}")
    model = fit_model(
        prices=prices, market=market, config=config,
        horizon=HORIZON, train_end=TRAIN_END,
        forest_params=FOREST_PARAMS, workers=WORKERS,
    )
    save_model(model, MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")


if __name__ == "__main__":
    main()
