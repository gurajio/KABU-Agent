import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = Path("logs/rf_test_20261004_203331_808788/predictions.csv")


def plot_predictions(path, bins=50):
    if bins < 1:
        raise ValueError("区間数は1以上にしてください。")
    frame = pd.read_csv(path)
    if "up_probability" not in frame:
        raise ValueError("CSVにup_probability列がありません。")
    values = pd.to_numeric(frame["up_probability"], errors="raise")
    missing = int(values.isna().sum())
    values = values.dropna()
    if values.empty:
        raise ValueError("可視化できる上昇確率がありません。")
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        raise ValueError("上昇確率には0〜1の有限値が必要です。")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.hist(values, bins=np.linspace(0, 1, bins + 1), color="#2563eb", edgecolor="white")
    ax.set(xlim=(0, 1), xlabel="Predicted probability of price increase", ylabel="Count",
           title=f"Predicted probability distribution (all symbols)\n{Path(path).parent.name}")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.text(0.98, 0.95, f"Valid: {len(values):,}\nMissing: {missing:,}",
            transform=ax.transAxes, ha="right", va="top")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    print(f"集計件数: {len(values):,}件 / 空欄の除外: {missing:,}件 / 区間幅: {100 / bins:g}%")
    return fig


def main():
    parser = argparse.ArgumentParser(description="上昇確率の度数分布を表示・PNG保存します。")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--no-show", action="store_true", help="ウィンドウを表示せず保存のみ行う")
    args = parser.parse_args()
    path = (ROOT / args.csv).resolve()
    fig = plot_predictions(path, args.bins)
    output = path.parent / f"predictions_histogram_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    with output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=150)
    print(f"保存先: {output}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
