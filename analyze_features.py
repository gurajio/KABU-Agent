import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from ml import load_model
from ml.models import build_samples
from train import ROOT, TRAIN_START, TRAIN_END, PERIODS, WORKERS, data_hash, load_data, score_classifier


def main():
    parser = argparse.ArgumentParser(description="検証期間で特徴量の相関と予測への寄与を調べます。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats は2以上にしてください。")
    model_path = (ROOT / args.model).resolve()
    model = load_model(model_path)
    fingerprint = data_hash()
    start, end = PERIODS["validation"]
    if (model.data_hash != fingerprint or model.train_start != TRAIN_START
            or model.train_end != pd.Timestamp(TRAIN_END) or model.train_end >= pd.Timestamp(start)):
        raise ValueError("モデルとCSV・学習期間が一致しません。")
    prices, market = load_data(end=end)
    if set(model.symbols) != set(prices):
        raise ValueError("モデルと対象銘柄が一致しません。")
    features, target = build_samples(prices, market, model.config, horizon=model.horizon, start=start, end=end)
    if features.empty or target.nunique() != 2:
        raise ValueError("検証には上昇・非上昇の両方の正解が必要です。")
    model.classifier.set_params(n_jobs=WORKERS)
    print(f"検証行数={len(features)}、予測使用コア数={WORKERS}、シャッフル反復={args.repeats}", flush=True)
    scores, _ = score_classifier(model.classifier, features, target, features, target)
    pearson, spearman = features.corr(), features.corr(method="spearman")
    with threadpool_limits(limits=1):
        importance = permutation_importance(
            model.classifier, features, target,
            scoring={"accuracy": "accuracy", "auc": "roc_auc", "brier": "neg_brier_score"},
            n_repeats=args.repeats, random_state=43, n_jobs=1,
        )
    rows = pd.DataFrame(index=features.columns)
    rows.index.name = "feature"
    for metric, result in importance.items():
        rows[f"{metric}_drop_mean"] = result.importances_mean
        rows[f"{metric}_drop_std"] = result.importances_std
    rows = rows.sort_values("auc_drop_mean", ascending=False)
    pairs = []
    for i, left in enumerate(features.columns):
        for right in features.columns[i + 1:]:
            pairs.append(dict(left=left, right=right, pearson=pearson.at[left, right],
                              spearman=spearman.at[left, right]))
    pairs = pd.DataFrame(pairs).sort_values("spearman", key=lambda x: x.abs(), ascending=False)
    if data_hash() != fingerprint:
        raise ValueError("分析中にCSVが変更されました。")
    output = ROOT / "logs" / f"feature_analysis_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    pearson.to_csv(output / "pearson.csv")
    spearman.to_csv(output / "spearman.csv")
    pairs.to_csv(output / "correlation_pairs.csv", index=False)
    rows.to_csv(output / "permutation_importance.csv")
    settings = {
        "model_path": str(model_path), "feature_names": list(features.columns),
        "validation_start": start, "validation_end": end, "validation_rows": len(features),
        "accuracy": scores["validation_accuracy"], "auc": scores["validation_auc"],
        "brier": scores["validation_brier"], "repeats": args.repeats, "seed": 43,
        "data_hash": fingerprint, "test_used": False, "model_adopted": False,
        "method": "検証行の1列を全銘柄・時刻間でシャッフル。正答率・AUCの低下、Brierの増加を寄与とする。",
        "limitations": "相関0は独立を保証しない。相関した入力の寄与は過小評価されうる。"
                        "シャッフルは時系列や指標間の整合を壊す診断であり、因果効果ではない。"
                        "反復標準偏差はシャッフルのばらつきであり、未知期間での信頼区間ではない。",
    }
    (output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(11, 9))
    mesh = axis.imshow(spearman, vmin=-1, vmax=1, cmap="RdBu_r")
    axis.set_xticks(range(len(features.columns)), features.columns, rotation=65, ha="right")
    axis.set_yticks(range(len(features.columns)), features.columns)
    axis.set_title("Validation feature correlation (Spearman)")
    for i in range(len(features.columns)):
        for j in range(len(features.columns)):
            axis.text(j, i, f"{spearman.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(mesh, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "correlation.png", dpi=150)
    plt.close(figure)
    print("相関が強い組み合わせ:\n" + pairs.head(8).to_string(index=False), flush=True)
    print("指標を崩した場合の性能低下:\n" + rows.to_string(), flush=True)
    print(f"保存先: {output}", flush=True)


if __name__ == "__main__":
    main()
