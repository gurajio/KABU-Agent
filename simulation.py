import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from functions import validate_order, update_portfolio, get_valuation_data, save_portfolio
from ml import FeatureConfig, load_model, predict_up
from strategy import random_strategy, rf_strategy
from create_log import create_equity_log, create_trade_log, save_logs
from train import ROOT, MODEL_PATH, TRAIN_START, TRAIN_END, PERIODS, FEATURE_PARAMS, HORIZON, load_data

INITIAL_CASH = 1_000_000
SEED = 43


def trade_times(prices, start, end):
    times = pd.DatetimeIndex(sorted({stamp for frame in prices.values() for stamp in frame.index}))
    minutes = times.hour * 60 + times.minute
    session = ((minutes >= 9 * 60) & (minutes < 11 * 60 + 30)) | (
        (minutes >= 12 * 60 + 30) & (minutes < 15 * 60 + 30)
    )
    return times[(times >= start) & (times + pd.Timedelta(minutes=5) <= end) & session]


def simulate(prices, probabilities, *, strategy, start, end, buy_threshold, sell_threshold):
    portfolio = {"cash": INITIAL_CASH, "positions": {}}
    rng = random.Random(SEED)
    trade_logs, equity_logs = [], []
    sold_today = set()
    simulation_date = None
    skipped = 0
    times = trade_times(prices, start, end)
    if times.empty:
        raise ValueError("指定期間に売買対象の5分足がありません。")
    for i, current_time in enumerate(times):
        if current_time.date() != simulation_date:
            simulation_date = current_time.date()
            sold_today.clear()
        for symbol, frame in prices.items():
            if current_time not in frame.index:
                continue
            price = frame.at[current_time, "Open"]
            if pd.isna(price):
                skipped += 1
                continue
            probability = None
            if strategy == "rf":
                probability = probabilities[symbol].get(current_time, float("nan"))
                order = rf_strategy(
                    symbol, price, portfolio, probability,
                    buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                )
            else:
                current_open = frame.loc[[current_time], ["Open"]]
                order = random_strategy(current_time, symbol, current_open, portfolio, rng)
            order["price"] = float(order["price"])
            if not validate_order(portfolio, order, sold_today):
                continue
            portfolio = update_portfolio(portfolio, order, current_time)
            log = create_trade_log(order, portfolio, current_time)
            log["reason"] = order["reason"]
            if probability is not None:
                log["up_probability"] = float(probability)
            trade_logs.append(log)
            if order["action"] == "sell":
                sold_today.add(symbol)
        if i == len(times) - 1 or times[i + 1].date() != current_time.date():
            latest_prices, valuation_time = get_valuation_data(prices, portfolio, current_time)
            for symbol, value in latest_prices.items():
                if pd.isna(value["latest_price"]):
                    raise ValueError(f"{symbol}の{current_time.date()}の評価価格が欠損しています。")
            equity_logs.append(create_equity_log(portfolio, latest_prices, valuation_time))
    return portfolio, trade_logs, equity_logs, skipped


def main():
    parser = argparse.ArgumentParser(description="保存済みデータで手法・評価期間を切り替えて売買します。")
    parser.add_argument("--strategy", choices=("rf", "random"), default="rf")
    parser.add_argument("--period", choices=tuple(PERIODS), default="validation")
    parser.add_argument("--buy-threshold", type=float, default=0.55)
    parser.add_argument("--sell-threshold", type=float, default=0.45)
    args = parser.parse_args()
    if not 0 <= args.sell_threshold < args.buy_threshold <= 1:
        parser.error("閾値は 0 ≦ 売り < 買い ≦ 1 にしてください。")
    start, end = map(pd.Timestamp, PERIODS[args.period])
    model = None
    if args.strategy == "rf":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"先にtrain.pyを実行してください: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        if (model.train_end != pd.Timestamp(TRAIN_END) or model.train_end >= start
                or model.config != FeatureConfig(**FEATURE_PARAMS) or model.horizon != HORIZON):
            raise ValueError("モデルの学習条件が現在の設定と異なります。train.pyで再学習してください。")
    prices, market = load_data(end=end)
    print(f"手法: {args.strategy}、期間: {start.date()} 〜 {end.date()}、銘柄数: {len(prices)}")
    probabilities = {}
    if model is not None:
        if set(model.symbols) != set(prices):
            raise ValueError("モデルと現在の対象銘柄が異なります。train.pyで再学習してください。")
        print(f"RF予測: 使用コア数={model.classifier.n_jobs}、各時刻までの確定足から計算")
        for symbol, frame in prices.items():
            result = predict_up(model, frame, market, as_of=end)
            probabilities[symbol] = result.loc[(result.index >= start) & (result.index <= end)]
        count = sum(int(series.notna().sum()) for series in probabilities.values())
        print(f"予測できた銘柄・時刻の組数: {count}")
        if not count:
            raise ValueError("指定期間に有効な予測がありません。")
    portfolio, trades, equity, skipped = simulate(
        prices, probabilities, strategy=args.strategy, start=start, end=end,
        buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold,
    )
    output = ROOT / "logs" / f"{args.strategy}_{args.period}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    save_logs(trades, equity, output_dir=output)
    save_portfolio(portfolio, path=output / "portfolio.json")
    if probabilities:
        forecasts = pd.concat(probabilities, names=["symbol", "decision_time"]).rename("up_probability")
        forecasts.to_csv(output / "predictions.csv")
    metadata = {
        "strategy": args.strategy, "period": args.period, "start": str(start), "end": str(end),
        "train_start": TRAIN_START, "train_end": TRAIN_END, "symbols": list(prices),
        "initial_cash": INITIAL_CASH, "seed": SEED, "cost_per_order": 0,
        "quantity_per_order": 100, "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold, "missing_open_skips": skipped,
        "fill_rule": "直前までに確定した足で判断し、現在の足のOpenで約定。手数料・スリッページ0。",
        "final_assets": equity[-1]["total_assets"],
        "model_path": str(MODEL_PATH) if model is not None else None,
        "features": asdict(model.config) if model is not None else None,
    }
    with (output / "settings.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"売買件数: {len(trades)}、始値欠損による見送り: {skipped}件")
    print(f"最終資産: {equity[-1]['total_assets']:,.0f}円")
    print(f"保存先: {output}")


if __name__ == "__main__":
    main()
