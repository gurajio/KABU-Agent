import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "logs/rf_validation_20261006_143552_826902/predictions.csv"


def main():
    parser = argparse.ArgumentParser(description="予測確率の度数分布を画像とCSVに保存します。")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--bin-width", type=int, default=5, help="階級幅（パーセントポイント）")
    parser.add_argument("--show", action="store_true", help="グラフをウィンドウにも表示します")
    args = parser.parse_args()
    if args.bin_width < 1 or args.bin_width > 100 or 100 % args.bin_width:
        parser.error("--bin-width は100を割り切れる1〜100の整数にしてください。")
    path = (ROOT / args.predictions).resolve()
    frame = pd.read_csv(path)
    if "up_probability" not in frame.columns:
        raise ValueError("up_probability 列がありません。")
    values = pd.to_numeric(frame.up_probability, errors="raise").dropna()
    if values.empty:
        raise ValueError("欠損を除くと予測確率がありません。")
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        raise ValueError("予測確率は0〜1の有限値にしてください。")
    edges = np.arange(0, 101, args.bin_width)
    counts, _ = np.histogram(values.to_numpy() * 100, bins=edges)
    frequency = pd.DataFrame({
        "lower_percent": edges[:-1], "upper_percent": edges[1:],
        "count": counts, "percentage": counts / len(values) * 100,
    })
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = path.parent / f"probability_histogram_{stamp}.png"
    csv_path = path.parent / f"probability_frequency_{stamp}.csv"
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(values * 100, bins=edges, edgecolor="white", color="#3478ad")
    axis.set(xlabel="Predicted probability of an increase (%)", ylabel="Count",
             title=f"Prediction probability distribution (n={len(values):,})", xlim=(0, 100))
    axis.set_xticks(np.arange(0, 101, 10))
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(image_path, dpi=150)
    frequency.to_csv(csv_path, index=False)
    print(f"有効予測: {len(values):,}件、欠損: {len(frame) - len(values):,}件")
    print(f"画像: {image_path}")
    print(f"度数分布CSV: {csv_path}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
