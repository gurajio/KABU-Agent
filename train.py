import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from ml import FeatureConfig, ModelBundle, fit_model, save_model
from ml.models import build_samples, fit_classifier
from ml.features import feature_columns

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
    "feature_set": "momentum12",
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
FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 7,
    "min_samples_leaf": 20,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "class_weight": None,
    "random_state": 43,
}
MODEL_CANDIDATES = ((5, 1), (10, 1), (10, 20), (20, 20), (20, 100), (None, 20))


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


def score_classifier(classifier, train_x, train_y, valid_x, valid_y):
    if valid_x.empty:
        raise ValueError("検証に評価可能な正解がありません。")
    with threadpool_limits(limits=1):
        probability = classifier.predict_proba(valid_x)[:, 1]
        train_probability = classifier.predict_proba(train_x)[:, 1]
    prior = float(train_y.mean())
    scores = {
        "training_accuracy": float(np.mean((train_probability >= 0.5) == train_y.to_numpy())),
        "validation_accuracy": float(np.mean((probability >= 0.5) == valid_y.to_numpy())),
        "training_correct": int(np.sum((train_probability >= 0.5) == train_y.to_numpy())),
        "validation_correct": int(np.sum((probability >= 0.5) == valid_y.to_numpy())),
        "training_count": len(train_y), "validation_count": len(valid_y),
        "validation_majority_accuracy": float(max(valid_y.mean(), 1 - valid_y.mean())),
        "validation_auc": float(roc_auc_score(valid_y, probability)) if valid_y.nunique() == 2 else None,
        "validation_brier": float(brier_score_loss(valid_y, probability)),
        "validation_log_loss": float(log_loss(valid_y, probability, labels=[0, 1])),
        "training_auc": float(roc_auc_score(train_y, train_probability)),
        "constant_brier": float(brier_score_loss(valid_y, np.full(len(valid_y), prior))),
        "training_up_rate": prior, "validation_up_rate": float(valid_y.mean()),
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "probability_min": float(probability.min()), "probability_max": float(probability.max()),
        "probability_q05": float(np.quantile(probability, 0.05)),
        "probability_q95": float(np.quantile(probability, 0.95)),
    }
    predictions = pd.DataFrame({"target": valid_y, "up_probability": probability})
    return scores, predictions


def fit_progress(train_x, train_y, valid_x, valid_y, params, *, steps=(10, 20, 50, 100)):
    if not steps or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in steps):
        raise ValueError("木の本数は正の整数で指定してください。")
    if tuple(sorted(set(steps))) != tuple(steps):
        raise ValueError("木の本数は重複なしの昇順にしてください。")
    classifier = None
    history = []
    for count in steps:
        if classifier is None:
            classifier = fit_classifier(
                train_x, train_y, forest_params=dict(params, n_estimators=count, warm_start=True), workers=WORKERS,
            )
        else:
            classifier.set_params(n_estimators=count)
            with threadpool_limits(limits=1):
                classifier.fit(train_x, train_y)
        metrics, _ = score_classifier(classifier, train_x, train_y, valid_x, valid_y)
        history.append({"n_estimators": count, **metrics})
        print(f"木の本数={count:3d}、学習正答率={metrics['training_accuracy']:.2%}、"
              f"検証正答率={metrics['validation_accuracy']:.2%}、"
              f"検証正解={metrics['validation_correct']}/{metrics['validation_count']}", flush=True)
    classifier.set_params(warm_start=False)
    return classifier, pd.DataFrame(history)


def compare_models(prices, market, config, output, fingerprint, *, learning_curve=False):
    start, end = PERIODS["validation"]
    if pd.Timestamp(TRAIN_END) >= pd.Timestamp(start):
        raise ValueError("学習と検証の期間を分離してください。")
    print(f"モデル比較: 学習={TRAIN_START}〜{TRAIN_END}、検証={start}〜{end}、"
          f"特徴量={config.feature_set}（{len(feature_columns(config))}列）")
    train_x, train_y = build_samples(prices, market, config, horizon=HORIZON, end=TRAIN_END)
    valid_x, valid_y = build_samples(prices, market, config, horizon=HORIZON, start=start, end=end)
    if valid_x.empty or valid_y.nunique() != 2:
        raise ValueError("検証には上昇・非上昇の両方の正解が必要です。")
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "status": "running", "train_start": TRAIN_START, "train_end": TRAIN_END,
        "validation_start": start, "validation_end": end, "test_used": False,
        "train_rows": len(train_x), "validation_rows": len(valid_x),
        "symbols": sorted(prices), "data_hash": fingerprint, "workers": WORKERS,
        "candidates": [{"max_depth": depth, "min_samples_leaf": leaf} for depth, leaf in MODEL_CANDIDATES],
        "model_adopted": False,
        "learning_curve": learning_curve,
        "tree_steps": [10, 20, 50, 100] if learning_curve else None,
        "accuracy_threshold": 0.5,
        "feature_set": config.feature_set, "feature_names": list(feature_columns(config)),
        "feature_params": vars(config),
        "metrics": "AUCは大きいほど良い。Brier・log lossは小さいほど良い。確率の幅だけで選ばない。",
        "prediction_scope": "comparisonsの予測は、欠損のない特徴量と正解がそろう検証行だけ。",
        "data_quality": "保存済みCSV。期間網羅性・無約定と配信欠損の判別は未確認。",
    }
    path = output / "settings.json"
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    scores = []
    try:
        for depth, leaf in MODEL_CANDIDATES:
            name = f"depth_{depth if depth is not None else 'none'}_leaf_{leaf}"
            params = dict(FOREST_PARAMS, max_depth=depth, min_samples_leaf=leaf)
            if learning_curve:
                print(f"候補: {name}（木を追加しながら評価）", flush=True)
                classifier, history = fit_progress(train_x, train_y, valid_x, valid_y, params)
                history.to_csv(output / f"{name}_learning_curve.csv", index=False)
            else:
                classifier = fit_classifier(train_x, train_y, forest_params=params, workers=WORKERS)
            metrics, predictions = score_classifier(classifier, train_x, train_y, valid_x, valid_y)
            if data_hash() != fingerprint:
                raise ValueError("比較中にCSVが変更されました。")
            bundle = ModelBundle(
                classifier, config, HORIZON, pd.Timestamp(TRAIN_END), len(train_x),
                tuple(sorted(prices)), sklearn.__version__, data_hash=fingerprint, train_start=TRAIN_START,
                feature_names=feature_columns(config),
            )
            save_model(bundle, output / f"{name}.pkl")
            predictions.to_csv(output / f"{name}_predictions.csv")
            scores.append({"candidate": name, "max_depth": depth, "min_samples_leaf": leaf, **metrics})
            print(f"{name}: 学習正答率={metrics['training_accuracy']:.2%}、"
                  f"検証正答率={metrics['validation_accuracy']:.2%}、"
                  f"AUC={metrics['validation_auc']:.6f}、Brier={metrics['validation_brier']:.6f}、"
                  f"確率の標準偏差={metrics['probability_std']:.6f}")
        if data_hash() != fingerprint:
            raise ValueError("比較中にCSVが変更されました。")
        pd.DataFrame(scores).to_csv(output / "comparisons.csv", index=False)
        settings["status"] = "complete"
    except Exception as error:
        settings.update(status="failed", error=str(error), completed_candidates=len(scores))
        raise
    finally:
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"比較結果: {output}（採用モデルは変更していません）")


def compare_symbols(prices, market, config, output, fingerprint, params):
    start, end = PERIODS["validation"]
    train_x, train_y = build_samples(prices, market, config, horizon=HORIZON, end=TRAIN_END)
    valid_x, valid_y = build_samples(prices, market, config, horizon=HORIZON, start=start, end=end)
    if valid_x.empty:
        raise ValueError("検証に評価可能な行がありません。")
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(status="running", model_adopted=False, test_used=False,
                    train_start=TRAIN_START, train_end=TRAIN_END, validation_start=start,
                    validation_end=end, feature_params=vars(config), forest_params=params,
                    accuracy_threshold=0.5, workers=WORKERS, data_hash=fingerprint,
                    symbols=sorted(prices), validation_scope="既存の1検証期間。複数期間での再現性は未確認。")
    rows, selections, individuals = [], {}, {}
    path = output / "settings.json"
    try:
        common = fit_classifier(train_x, train_y, forest_params=params, workers=WORKERS)
        shared = ModelBundle(common, config, HORIZON, pd.Timestamp(TRAIN_END), len(train_x),
                             tuple(sorted(prices)), sklearn.__version__, feature_names=feature_columns(config),
                             data_hash=fingerprint, train_start=TRAIN_START)
        save_model(shared, output / "common.pkl")
        for number, symbol in enumerate(sorted(prices), 1):
            selections[symbol] = individuals[symbol] = "common.pkl"
            tx = train_x.loc[train_x.index.get_level_values("symbol") == symbol]
            ty = train_y.reindex(tx.index)
            vx = valid_x.loc[valid_x.index.get_level_values("symbol") == symbol]
            vy = valid_y.reindex(vx.index)
            row = dict(symbol=symbol, training_count=len(tx), validation_count=len(vx), status="skipped")
            if vx.empty:
                row["reason"] = "検証行なし"
                rows.append(row)
                continue
            if tx.empty or ty.nunique() != 2:
                row["reason"] = "個別モデルの学習に必要な2クラスがない"
                rows.append(row)
                continue
            baseline, predictions = score_classifier(common, tx, ty, vx, vy)
            print(f"[{number}/{len(prices)}] {symbol} 個別モデル学習", flush=True)
            classifier = fit_classifier(tx, ty, forest_params=params, workers=WORKERS)
            scores, individual_prediction = score_classifier(classifier, tx, ty, vx, vy)
            bundle = ModelBundle(classifier, config, HORIZON, pd.Timestamp(TRAIN_END), len(tx),
                                 (symbol,), sklearn.__version__, feature_names=feature_columns(config),
                                 data_hash=fingerprint, train_start=TRAIN_START)
            save_model(bundle, output / f"{symbol}.pkl")
            individuals[symbol] = f"{symbol}.pkl"
            predictions.rename(columns={"up_probability": "common_probability"}).assign(
                individual_probability=individual_prediction.up_probability
            ).to_csv(output / f"{symbol}_predictions.csv")
            row.update(status="complete", majority_accuracy=scores["validation_majority_accuracy"],
                       common_accuracy=baseline["validation_accuracy"], individual_accuracy=scores["validation_accuracy"],
                       accuracy_delta=scores["validation_accuracy"] - baseline["validation_accuracy"],
                       common_auc=baseline["validation_auc"], individual_auc=scores["validation_auc"],
                       common_brier=baseline["validation_brier"], individual_brier=scores["validation_brier"],
                       common_training_accuracy=baseline["training_accuracy"],
                       individual_training_accuracy=scores["training_accuracy"],
                       common_correct=baseline["validation_correct"], individual_correct=scores["validation_correct"])
            rows.append(row)
            print(f"{symbol}: 共通正答率={row['common_accuracy']:.2%}、"
                  f"個別正答率={row['individual_accuracy']:.2%}、差={row['accuracy_delta']:+.2%}", flush=True)
        if data_hash() != fingerprint:
            raise ValueError("比較中にCSVが変更されました。")
        for name, mapping in (("selection.json", selections), ("individual_models.json", individuals)):
            (output / name).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        settings.update(status="complete", trained_symbols=sum(r["status"] == "complete" for r in rows),
                        skipped_symbols=sum(r["status"] == "skipped" for r in rows))
    except Exception as error:
        settings.update(status="failed", error=str(error))
        raise
    finally:
        pd.DataFrame(rows).to_csv(output / "symbol_comparisons.csv", index=False)
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"保存先: {output}（採用モデルは変更していません）", flush=True)


def main():
    parser = argparse.ArgumentParser(description="保存済みCSVからRFモデルを学習します。")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--max-depth", type=int, default=FOREST_PARAMS["max_depth"])
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--compare", action="store_true", help="6候補を学習・検証し、採用せず別保存します")
    parser.add_argument("--per-symbol", action="store_true", help="共通モデルと銘柄別モデルを同じ検証行で比較します")
    parser.add_argument("--learning-curve", action="store_true", help="--compareで木10・20・50・100本時の正答率を出力します")
    parser.add_argument("--feature-set", choices=("legacy10", "momentum12"), default="momentum12")
    parser.add_argument("--output", type=Path, help="--compare または --per-symbol の新しい保存先")
    args = parser.parse_args()
    if args.per_symbol and (args.compare or args.learning_curve):
        parser.error("--per-symbol は --compare・--learning-curve と併用できません。")
    if args.learning_curve and not args.compare:
        parser.error("--learning-curve は --compare と一緒に指定してください。")
    if args.max_depth < 1 or args.min_samples_leaf < 1:
        parser.error("木の深さ・葉の最低標本数は1以上にしてください。")
    if args.output is not None and not (args.compare or args.per_symbol):
        parser.error("--output は --compare または --per-symbol と一緒に指定してください。")
    destination = (ROOT / args.model).resolve()
    config = FeatureConfig(**dict(FEATURE_PARAMS, feature_set=args.feature_set))
    if args.per_symbol:
        output = (ROOT / args.output).resolve() if args.output else (
            ROOT / "logs" / f"symbol_validation_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        if output.exists():
            raise FileExistsError(f"保存先が存在します: {output}")
        fingerprint = data_hash()
        prices, market = load_data(end=PERIODS["validation"][1])
        compare_symbols(prices, market, config, output, fingerprint,
                        dict(FOREST_PARAMS, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf))
        return
    if args.compare:
        output = (ROOT / args.output).resolve() if args.output else (
            ROOT / "logs" / f"model_validation_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        if output.exists():
            raise FileExistsError(f"保存先が存在します: {output}")
        fingerprint = data_hash()
        prices, market = load_data(end=PERIODS["validation"][1])
        compare_models(prices, market, config, output, fingerprint, learning_curve=args.learning_curve)
        return
    if destination.exists():
        raise FileExistsError(f"保存先が存在します。--modelで別名を指定してください: {destination}")
    fingerprint = data_hash()
    prices, market = load_data()
    print(f"学習期間: {TRAIN_START} 〜 {TRAIN_END}、銘柄数: {len(prices)}")
    model = fit_model(
        prices=prices, market=market, config=config,
        horizon=HORIZON, train_end=TRAIN_END,
        forest_params=dict(FOREST_PARAMS, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf),
        workers=WORKERS,
    )
    if data_hash() != fingerprint:
        raise ValueError("学習中にCSVが変更されたため、モデルを保存しません。")
    model.data_hash = fingerprint
    model.train_start = TRAIN_START
    save_model(model, destination)
    print(f"モデルを保存しました: {destination}")


if __name__ == "__main__":
    main()
