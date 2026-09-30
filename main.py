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
    get_save_latest_stock_data,
)
from strategy import random_strategy
from create_log import create_equity_log,create_trade_log,save_logs

# 乱数生成器を固定
FIXED_SEED = False
rng = random.Random(43 if FIXED_SEED else None)
# 銘柄一覧CSVを読み込む
stocks = pd.read_csv("tickers.csv")

# ポートフォリオを読み込む
with open("portfolio.json", "r", encoding="utf-8") as file:
    portfolio = json.load(file)

# 銘柄ごとのDataFrameを入れる辞書
stock_data = {}

sold_today = set()
simulation_date = None

# ログを作成しておく
trade_logs = []
equity_logs = []
# 実行ごとに保存先を変えて、前回のログも残す
log_dir = f"logs/"

# 実行した時
# 現時点での全銘柄のデータを取得
for symbol in stocks["symbol"]:
    latest = get_save_latest_stock_data(symbol)
    stock_data[symbol] = pd.DataFrame([latest]).set_index("Datetime")

# strategy.pyをもとに売買の判断を行う
for symbol, df in stock_data.items(): 
    current_time = df.index[-1]
    # 売買判断
    order = random_strategy(current_time, symbol, df, portfolio, rng)
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

    if stock_data:
        evaluation_time = max(df.index[-1] for df in stock_data.values())
        latest_prices, valuation_time = get_valuation_data(
            stock_data, portfolio, evaluation_time
        )
        equity_logs.append(
            create_equity_log(portfolio, latest_prices, valuation_time)
        )

# シミュレーション終了後の口座状況を保存する
save_portfolio(portfolio)
# 今回の全売買履歴と各日の資産ログを、終了後にまとめて保存する
save_logs(trade_logs, equity_logs, output_dir=log_dir)
print(f"ログの保存先: {log_dir}/simulation.json")

print(portfolio)
