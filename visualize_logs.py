import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter, MaxNLocator


# このPythonファイルと同じ場所にあるlogsフォルダを読み込む
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FOLDER = "rf_validation_20261005_155058_944117"


def load_logs(log_dir):
    # JSONのリストをDataFrameへ変換する
    with open(log_dir / "trade_logs.json", encoding="utf-8") as file:
        trades = pd.DataFrame(json.load(file))

    with open(log_dir / "equity_logs.json", encoding="utf-8") as file:
        equity = pd.DataFrame(json.load(file))

    if equity.empty:
        raise ValueError("資産ログが空です。先にシミュレーションを実行してください。")

    equity["datetime"] = pd.to_datetime(equity["datetime"], utc=True).dt.tz_convert("Asia/Tokyo")
    equity = equity.sort_values("datetime", kind="stable")
    equity = equity.drop_duplicates(subset="datetime", keep="last")
    equity = equity.groupby(equity["datetime"].dt.normalize(), sort=True).tail(1)

    if not trades.empty:
        trades["datetime"] = pd.to_datetime(trades["datetime"], utc=True).dt.tz_convert("Asia/Tokyo")
        trades = trades.sort_values("datetime", kind="stable")

    return trades, equity


def plot_logs(trades, equity, *, title=""):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    if title:
        fig.suptitle(title)

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


def main():
    parser = argparse.ArgumentParser(description="指定ログフォルダから日単位の資産推移をPNGに保存します。")
    parser.add_argument("folder", nargs="?", type=Path, default=Path(LOG_FOLDER), help="省略時はコード上部のLOG_FOLDERを使用")
    parser.add_argument("--show", action="store_true", help="画像保存後にグラフのウィンドウも表示する")
    args = parser.parse_args()
    if args.folder.is_absolute():
        log_dir = args.folder
    elif args.folder.parts[0] == "logs":
        log_dir = LOG_DIR.parent / args.folder
    else:
        log_dir = LOG_DIR / args.folder
    log_dir = log_dir.resolve()
    try:
        trades, equity = load_logs(log_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"ログを読み込めません: {log_dir}\n{error}")
    fig = plot_logs(trades, equity, title=log_dir.name)

    output_path = log_dir / f"visualization_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"売買履歴: {len(trades):,}件 / 資産ログ: {len(equity)}件")
    print(f"最終評価日時: {equity.iloc[-1]['datetime']}")
    print(f"最終総資産: {equity.iloc[-1]['total_assets']:,.0f}円")
    if not trades.empty:
        print("\n直近10件の売買履歴:")
        print(trades.tail(10).to_string(index=False))
    print(f"\nグラフの保存先: {output_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
