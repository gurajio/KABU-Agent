# yfinanceのテストを行う
# 実際にデータをpython経由で取得するためのライブラリ

import yfinance as yf
import pandas as pd

toyota_ticker_code = '7203.T'
toyota = yf.Ticker(toyota_ticker_code)

df_toyota = toyota.history(start='2024-01-01',end='2024-06-03')

print(df_toyota.head())

