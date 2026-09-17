import pandas as pd
import yfinance as yf
from pathlib import Path

def get_save_stock_data(code, reflesh = False):
    # 銘柄コードに対応する株価データを取得・保存する。
    # 初回はデータを取得し、dataフォルダ内に「銘柄名_5m.csv」という形で保存される、2回目以降はそのままdfで出力、プロパティrefleshを指定したら最新の時間で取得する
    # 現状の設定としては「5分足」「取得可能な全期間」を取得
    # 内容としては「始値、高値、安値、終値、出来高」
    # 入力：銘柄コード(string)、出力：DataFrame
    ticker = yf.Ticker(code)
    
    path = Path("data")/f"{code}_5m.csv"
    if path.exists() and not reflesh:
        return pd.read_csv(path, index_col=0, parse_dates=[0])
    
    df = ticker.history(
        period="max",
        interval="5m",
        auto_adjust=False,
    )
    
    if df.empty:
        raise ValueError(f"{code} の株価を取得できませんでした")
    
    df = df[["Open","High","Low","Close","Volume"]]
    
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path)
    
    return df

def get_all_timestamps(all_data):
    # すべての銘柄をまとめたデータ（辞書型）を用いてすべてのタイムスタンプを取得する
    all_times = []
    
    for df in all_data.values():
        all_times.extend(df.index)
    
    all_times = sorted(set(all_times))
    
    return all_times


def validate_order(portfolio, aciton):
    