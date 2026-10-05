import argparse
import json
import random
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from functions import order_reason, update_portfolio, get_valuation_data, save_portfolio
from ml import FeatureConfig, load_model, predict_up
from strategy import random_strategy, rf_strategy
from create_log import create_equity_log, create_trade_log, save_logs
from train import ROOT, MODEL_PATH, TRAIN_START, TRAIN_END, PERIODS, FEATURE_PARAMS, HORIZON, load_data, data_hash

INITIAL_CASH = 100_000_000
SEED = 43
CASH_RESERVE = 0.10
POSITION_LIMIT = 0.10

def trade_times(prices, start, end):
    times = pd.DatetimeIndex(sorted({stamp for frame in prices.values() for stamp in frame.index}))
    minutes = times.hour * 60 + times.minute
    session = ((minutes >= 9 * 60) & (minutes < 11 * 60 + 30)) | (
        (minutes >= 12 * 60 + 30) & (minutes < 15 * 60 + 30)
    )
    return times[(times >= start) & (times + pd.Timedelta(minutes=5) <= end) & session]


def decision_prices(prices, times):
    marks = {}
    for symbol, frame in prices.items():
        close = frame.Close.dropna().copy()
        close.index = close.index + pd.Timedelta(minutes=5)
        past = close.reindex(times, method="ffill")
        marks[symbol] = frame.Open.reindex(times).combine_first(past)
    return pd.DataFrame(marks, index=times)


def simulate(prices, probabilities, *, strategy, start, end, buy_threshold, sell_threshold,
             initial_cash=INITIAL_CASH, reserve_ratio=CASH_RESERVE,
             max_position_ratio=POSITION_LIMIT, diagnostics=None):
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("初期資金は正の有限値にしてください。")
    if not math.isfinite(reserve_ratio) or not 0 <= reserve_ratio < 1:
        raise ValueError("予備資金の割合は0以上1未満にしてください。")
    if not math.isfinite(max_position_ratio) or not 0 < max_position_ratio <= 1:
        raise ValueError("1銘柄の配分上限は0より大きく1以下にしてください。")
    portfolio = {"cash": initial_cash, "positions": {}}
    rng = random.Random(SEED)
    trade_logs, equity_logs = [], []
    sold_today = set()
    simulation_date = None
    skipped = 0
    counts = Counter()
    times = trade_times(prices, start, end)
    if times.empty:
        raise ValueError("指定期間に売買対象の5分足がありません。")
    marks = decision_prices(prices, times)
    for i, current_time in enumerate(times):
        latest_prices = marks.iloc[i].to_dict()
        if current_time.date() != simulation_date:
            simulation_date = current_time.date()
            sold_today.clear()
        for symbol, frame in prices.items():
            if current_time not in frame.index:
                continue
            price = frame.at[current_time, "Open"]
            if pd.isna(price):
                skipped += 1
                counts["missing_open"] += 1
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
            if order["action"] == "keep" and probability is not None and pd.isna(probability):
                counts["missing_prediction"] += 1
                continue
            reason = order_reason(
                portfolio, order, sold_today, latest_prices=latest_prices,
                reserve_ratio=reserve_ratio, max_position_ratio=max_position_ratio,
            )
            if reason is not None:
                counts[reason] += 1
                continue
            portfolio = update_portfolio(portfolio, order, current_time)
            counts[f"{order['action']}_filled"] += 1
            log = create_trade_log(order, portfolio, current_time)
            assets = portfolio["cash"] + sum(
                held["quantity"] * latest_prices[code] for code, held in portfolio["positions"].items()
            )
            log["decision_assets"] = float(assets)
            log["cash_ratio_after"] = float(portfolio["cash"] / assets)
            log["position_ratio_after"] = float(
                portfolio["positions"].get(symbol, {}).get("quantity", 0) * price / assets
            )
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
    if diagnostics is not None:
        diagnostics.update(counts)
    return portfolio, trade_logs, equity_logs, skipped


def main():
    parser = argparse.ArgumentParser(description="保存済みデータで手法・評価期間を切り替えて売買します。")
    parser.add_argument("--strategy", choices=("rf", "random"), default="rf")
    parser.add_argument("--period", choices=tuple(PERIODS), default="validation")
    parser.add_argument("--buy-threshold", type=float, default=0.55)
    parser.add_argument("--sell-threshold", type=float, default=0.45)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--initial-cash", type=float, default=INITIAL_CASH)
    parser.add_argument("--cash-reserve", type=float, default=CASH_RESERVE)
    parser.add_argument("--position-limit", type=float, default=POSITION_LIMIT)
    args = parser.parse_args()
    if not 0 <= args.sell_threshold < args.buy_threshold <= 1:
        parser.error("閾値は 0 ≦ 売り < 買い ≦ 1 にしてください。")
    if not math.isfinite(args.initial_cash) or args.initial_cash <= 0:
        parser.error("初期資金は正の有限値にしてください。")
    if not math.isfinite(args.cash_reserve) or not 0 <= args.cash_reserve < 1:
        parser.error("予備資金の割合は0以上1未満にしてください。")
    if not math.isfinite(args.position_limit) or not 0 < args.position_limit <= 1:
        parser.error("1銘柄の配分上限は0より大きく1以下にしてください。")
    start, end = map(pd.Timestamp, PERIODS[args.period])
    model = None
    model_path = (ROOT / args.model).resolve()
    fingerprint = data_hash()
    if args.strategy == "rf":
        if not model_path.exists():
            raise FileNotFoundError(f"先にtrain.pyを実行してください: {model_path}")
        model = load_model(model_path)
        if getattr(model, "data_hash", None) != fingerprint or getattr(model, "train_start", None) != TRAIN_START:
            raise ValueError("モデルと現在のCSV・学習開始日が一致しません。train.pyで新しいモデルを作成してください。")
        if (model.train_end != pd.Timestamp(TRAIN_END) or model.train_end >= start
                or model.config != FeatureConfig(**FEATURE_PARAMS) or model.horizon != HORIZON):
            raise ValueError("モデルの学習条件が現在の設定と異なります。train.pyで再学習してください。")
    prices, market = load_data(end=end)
    print(f"手法: {args.strategy}、期間: {start.date()} 〜 {end.date()}、銘柄数: {len(prices)}")
    print(f"初期資金: {args.initial_cash:,.0f}円、予備資金: {args.cash_reserve:.0%}、"
          f"1銘柄上限: {args.position_limit:.0%}（購入時の総資産に対する割合）")
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
    diagnostics = {}
    portfolio, trades, equity, skipped = simulate(
        prices, probabilities, strategy=args.strategy, start=start, end=end,
        buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold,
        initial_cash=args.initial_cash, reserve_ratio=args.cash_reserve,
        max_position_ratio=args.position_limit, diagnostics=diagnostics,
    )
    if data_hash() != fingerprint:
        raise ValueError("実行中にCSVが変更されたため、シミュレーション結果を保存しません。")
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
        "initial_cash": args.initial_cash, "seed": SEED, "cost_per_order": 0,
        "quantity_per_order": 100, "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold, "missing_open_skips": skipped,
        "cash_reserve": args.cash_reserve, "position_limit": args.position_limit,
        "order_counts": diagnostics,
        "limit_rule": "購入直前の現在足のOpen、なければ過去の確定Closeで総資産を評価。"
                      "手数料控除後の総資産で予備資金・既存保有と買い増しを含む銘柄上限を判定。"
                      "価格変動による超過では強制売却しない。",
        "fill_rule": "直前までに確定した足で判断し、現在の足のOpenで約定。手数料・スリッページ0。",
        "final_assets": equity[-1]["total_assets"],
        "model_path": str(model_path) if model is not None else None,
        "data_hash": fingerprint,
        "data_quality": "欠損は補完せず、特徴量不足では判断なし。取得条件・期間網羅性は未確認。",
        "features": asdict(model.config) if model is not None else None,
    }
    with (output / "settings.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"売買件数: {len(trades)}、始値欠損による見送り: {skipped}件")
    print(f"注文集計: {diagnostics}")
    print(f"最終資産: {equity[-1]['total_assets']:,.0f}円")
    print(f"保存先: {output}")


if __name__ == "__main__":
    main()
