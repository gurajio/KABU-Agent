import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter, MaxNLocator


# このPythonファイルと同じ場所にあるlogsフォルダを読み込む
LOG_DIR = Path(__file__).resolve().parent / "logs"


def load_logs(log_dir):
    # JSONのリストをDataFrameへ変換する
    with open(log_dir / "trade_logs.json", encoding="utf-8") as file:
        trades = pd.DataFrame(json.load(file))

    with open(log_dir / "equity_logs.json", encoding="utf-8") as file:
        equity = pd.DataFrame(json.load(file))

    if equity.empty:
        raise ValueError("資産ログが空です。先にシミュレーションを実行してください。")

    equity["datetime"] = pd.to_datetime(equity["datetime"], utc=True).dt.tz_convert("Asia/Tokyo")
    equity = equity.sort_values("datetime")

    if not trades.empty:
        trades["datetime"] = pd.to_datetime(trades["datetime"], utc=True).dt.tz_convert("Asia/Tokyo")
        trades = trades.sort_values("datetime", kind="stable")

    return trades, equity


def plot_logs(trades, equity):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")

    # 上段：総資産・現金・保有株の評価額
    dates = equity["datetime"].dt.normalize()
    stock_value = equity["total_assets"] - equity["cash"]
    axes[0].plot(dates, equity["total_assets"], label="Total assets", color="#2563eb", linewidth=2)
    axes[0].plot(dates, equity["cash"], label="Cash", color="#64748b")
    axes[0].plot(dates, stock_value, label="Stock value", color="#0d9488")
    axes[0].set_title("Portfolio value at each recorded daily valuation")
    axes[0].set_ylabel("JPY")
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1))

    # 下段：1日あたりの買い・売りの成立回数（株数ではなく注文の件数）
    days = pd.DatetimeIndex(dates.unique()).sort_values()
    daily_counts = pd.DataFrame(0, index=days, columns=["buy", "sell"])

    if not trades.empty:
        daily_counts = (
            trades.groupby([trades["datetime"].dt.normalize(), "action"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=days, columns=["buy", "sell"], fill_value=0)
        )

    axes[1].bar(days, daily_counts["buy"], label="Buy", color="#2563eb")
    axes[1].bar(days, daily_counts["sell"], bottom=daily_counts["buy"], label="Sell", color="#f97316")
    axes[1].set_title("Daily executed trades")
    axes[1].set_ylabel("Number of trades")
    axes[1].set_xlabel("Date (JST)")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))

    timezone = ZoneInfo("Asia/Tokyo")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8, tz=timezone)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone))

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    return fig


if __name__ == "__main__":
    trades, equity = load_logs(LOG_DIR)
    fig = plot_logs(trades, equity)

    # 画像としても保存し、グラフをウィンドウで表示する
    output_path = LOG_DIR / "visualization.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"売買履歴: {len(trades):,}件 / 資産ログ: {len(equity)}件")
    print(f"最終評価日時: {equity.iloc[-1]['datetime']}")
    print(f"最終総資産: {equity.iloc[-1]['total_assets']:,.0f}円")
    if not trades.empty:
        print("\n直近10件の売買履歴:")
        print(trades.tail(10).to_string(index=False))
    print(f"\nグラフの保存先: {output_path}")
    plt.show()
