import pandas as pd
import matplotlib.pyplot as plt
import random
import json
from datetime import datetime

from functions import (
    get_save_stock_data,
    get_all_timestamps,
    validate_order,
    update_portfolio,
    save_portfolio,
    get_valuation_data,
)
from strategy import random_strategy
from create_log import create_equity_log,create_trade_log,save_logs

# 乱数生成器を固定
rng = random.Random(43)
# 銘柄一覧CSVを読み込む
stocks = pd.read_csv("tickers.csv")

# ポートフォリオを読み込む
with open("portfolio.json", "r", encoding="utf-8") as file:
    portfolio = json.load(file)

# 銘柄ごとのDataFrameを入れる辞書
stock_data = {}

sold_today = set()
simulation_date = None
# 銘柄ごとの株価情報を取得
for symbol in stocks["symbol"]:
    df = get_save_stock_data(symbol,reflesh=False)
    stock_data[symbol] = df
# ログを作成しておく
trade_logs = []
equity_logs = []
# 実行ごとに保存先を変えて、前回のログも残す
log_dir = f"logs/"

# シミュレータを実行
# 時刻列を作成（全銘柄のタイムスタンプを取得し、古い順に並べ替え、重複削除）
all_times = get_all_timestamps(stock_data)

for i, current_time in enumerate(all_times):
    # 差金決済対策のための1日あたりの売り注文ログのクリア（同じ日に同じものを占いようにするためんい）
    if current_time.date() != simulation_date:
        simulation_date = current_time.date()
        sold_today.clear()
    for symbol, df in stock_data.items():
        # この銘柄に、その時刻のデータがなければ飛ばす
        if current_time not in df.index:
            continue
        row = df.loc[current_time]
        # print(current_time, symbol, row["Open"])
        # 売買判断
        order = random_strategy(current_time, symbol, stock_data[symbol], portfolio, rng)
        # validate_orderにsold_todayを渡して確認
        if not validate_order(portfolio, order, sold_today):
            continue
        # 売買成立・portfolio更新
        # 注文を確認する
        portfolio = update_portfolio(portfolio, order, current_time)
        # 売買履歴ログを作成
        trade_logs.append(create_trade_log(order, portfolio, current_time))

        # 売却が成立した銘柄を記録する
        if order["action"] == "sell":
            sold_today.add(order["symbol"])

        print(portfolio)

    # その時刻の全銘柄を処理した後、1日につき1件の資産ログを追加する
    is_day_end = (
        i == len(all_times) - 1
        or all_times[i + 1].date() != current_time.date()
    )

    if is_day_end:
        latest_prices, valuation_time = get_valuation_data(
            stock_data, portfolio, current_time
        )

        equity_logs.append(
            create_equity_log(
                portfolio, latest_prices, valuation_time
            )
        )

# シミュレーション終了後の口座状況を保存する
save_portfolio(portfolio)
# 今回の全売買履歴と各日の資産ログを、終了後にまとめて保存する
save_logs(trade_logs, equity_logs, output_dir=log_dir)
print(f"ログの保存先: {log_dir}/simulation.json")

print(portfolio)
