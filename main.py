import pandas as pd
import matplotlib.pyplot as plt
import random
import json

from functions import get_save_stock_data, get_all_timestamps
from strategy import random_strategy

# 乱数生成器を固定
rng = random.Random(43)
# 銘柄一覧CSVを読み込む
stocks = pd.read_csv("tickers.csv")

# ポートフォリオを読み込む
with open("portfolio.json", "r", encoding="utf-8") as file:
    portfolio = json.load(file)
    

# 銘柄ごとのDataFrameを入れる辞書
stock_data = {}

for symbol in stocks["symbol"]:
    df = get_save_stock_data(symbol)
    stock_data[symbol] = df

# 指定した銘柄のDataFrameを表示する
# print(stock_data["9432.T"].index[0])
# print(stock_data["9432.T"].iloc[0])

# シミュレータを実行
# 時刻列を作成（全銘柄のタイムスタンプを取得し、古い順に並べ替え、重複削除）
all_times = get_all_timestamps(stock_data)
for current_time in all_times[:2]:
    for symbol, df in stock_data.items():
        # この銘柄に、その時刻のデータがなければ飛ばす
        if current_time not in df.index:
            continue

        row = df.loc[current_time]

        # print(current_time, symbol, row["Open"])
        
        action = random_strategy(current_time, symbol, stock_data[symbol], portfolio, rng)
        print(action)
        
        