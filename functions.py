import pandas as pd
import yfinance as yf
from pathlib import Path
import json

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


def validate_order(portfolio, order, sold_today):
    symbol = order["symbol"]
    quantity = order["quantity"]
    price = order["price"]
    cost = order["cost"]

    position = portfolio["positions"].get(symbol, {})
    holding_quantity = position.get("quantity", 0)
    cash = portfolio["cash"]

    if order["action"] == "buy":
        # 当日売却した銘柄は再購入しない
        if symbol in sold_today:
            return False

        required_cash = price * quantity + cost
        return cash >= required_cash

    elif order["action"] == "sell":
        # 保有数量を超えて売れない
        if quantity > holding_quantity:
            return False

        # 売却後の現金を確認する
        return cash + price * quantity - cost >= 0

    # keepなど、売買しない場合
    return False

def update_portfolio(portfolio, order, current_time):
    symbol = order["symbol"]
    price = order["price"]
    quantity = order["quantity"]
    cost = order["cost"]

    positions = portfolio["positions"]

    if order["action"] == "buy":
        # 初めて購入する銘柄
        if symbol not in positions:
            positions[symbol] = {
                "quantity": quantity,
                "entry_price": price,
                "entry_time": str(current_time),
            }

        # すでに保有している銘柄の買い増し
        else:
            position = positions[symbol]

            new_quantity = position["quantity"] + quantity

            position["entry_price"] = (
                position["entry_price"] * position["quantity"]
                + price * quantity
            ) / new_quantity

            position["quantity"] = new_quantity

        # 購入代金とコストを差し引く
        portfolio["cash"] -= price * quantity + cost

    elif order["action"] == "sell":
        position = positions[symbol]

        # 保有数量を減らす
        position["quantity"] -= quantity

        # 売却代金からコストを引いて受け取る
        portfolio["cash"] += price * quantity - cost

        # 全株売却したら保有情報を削除する
        if position["quantity"] == 0:
            del positions[symbol]

    return portfolio


def save_portfolio(portfolio, path="portfolio.json"):
    # 現在の口座状況をJSONファイルに上書き保存する
    with open(path, "w", encoding="utf-8") as file:
        json.dump(portfolio, file, ensure_ascii=False, indent=4)

def get_valuation_data(stock_data, portfolio, current_time):
    # 5分足の開始時刻に5分を加え、足が終了する時刻を評価日時にする
    valuation_time = current_time + pd.Timedelta(minutes=5)

    latest_prices = {}

    # 保有している銘柄だけ評価価格を取り出す
    for symbol in portfolio["positions"]:
        df = stock_data[symbol]

        # 同じ日の、処理済みの時刻までのデータに絞る
        day_data = df[
            (df.index.date == current_time.date())
            & (df.index <= current_time)
        ]

        if day_data.empty:
            raise ValueError(
                f"{symbol} の {current_time.date()} の評価価格がありません"
            )

        # その銘柄の最後の行から終値を取得する
        last_row = day_data.iloc[-1]

        latest_prices[symbol] = {
            "latest_price": float(last_row["Close"]),
            "price_datetime": last_row.name,
        }

    return latest_prices, valuation_time