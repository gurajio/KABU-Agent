import pandas as pd
from functions import get_save_stock_data
import matplotlib.pyplot as plt

# 銘柄一覧CSVを読み込む
stocks = pd.read_csv("tickers.csv")

# 銘柄ごとのDataFrameを入れる辞書
stock_data = {}

for symbol in stocks["symbol"]:
    df = get_save_stock_data(symbol)
    stock_data[symbol] = df

# 指定した銘柄のDataFrameを表示する
print(stock_data["9432.T"])

# グラフにする銘柄を指定
symbol = "9501.T"
df = stock_data[symbol]

# グラフを作成
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Open"])

# タイトル・軸の説明
plt.title(f"{symbol} - Open Price")
plt.xlabel("Date")
plt.ylabel("Price (JPY)")

# 表示を整える
plt.grid(True)
plt.tight_layout()

# グラフを表示
plt.show()