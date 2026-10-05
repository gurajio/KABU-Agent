import pandas as pd
import yfinance as yf
from pathlib import Path
import json
import os
import tempfile
import math

def get_save_latest_stock_data(code: str) -> dict:
    path = Path("data") / f"{code}_5m.csv"
    df = yf.Ticker(code).history(
        period="1d", interval="5m", auto_adjust=False
    )
    if df.empty:
        raise ValueError(f"{code} の株価を取得できませんでした")

    latest = df[["Open", "High", "Low", "Close", "Volume"]].tail(1)
    result = {"Datetime": latest.index[0], **latest.iloc[0].to_dict()}
    has_data = path.exists() and path.stat().st_size > 0

    if has_data:
        saved = pd.read_csv(path, usecols=["Datetime"])
        saved_times = pd.to_datetime(saved["Datetime"], utc=True)
        latest_time = pd.to_datetime(latest.index[0], utc=True)
        if saved_times.eq(latest_time).any():
            return result

    path.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(
        path,
        mode="a",
        header=not has_data,
        index_label="Datetime",
    )
    return result

def get_save_stock_data(code, reflesh = False):
    # 銘柄コードに対応する株価データを取得・保存する。
    # 初回はデータを取得し、dataフォルダ内に「銘柄名_5m.csv」という形で保存される、2回目以降はそのままdfで出力、プロパティrefleshを指定したら最新の時間で取得する
    # 現状の設定としては「5分足」「取得可能な全期間」を取得
    # 内容としては「始値、高値、安値、終値、出来高」
    # 入力：銘柄コード(string)、出力：DataFrame
    path = Path("data")/f"{code}_5m.csv"
    saved = None
    if path.exists():
        saved = pd.read_csv(path, index_col=0, parse_dates=[0])
        saved.index = saved.index.tz_convert("Asia/Tokyo")
        saved = saved.loc[~saved.index.duplicated(keep="last")].sort_index()
        if not reflesh:
            return saved
    
    df = yf.Ticker(code).history(
        period="max",
        interval="5m",
        auto_adjust=False,
    )
    
    if df.empty:
        raise ValueError(f"{code} の株価を取得できませんでした")
    
    df = df[["Open","High","Low","Close","Volume"]]
    df.index = df.index.tz_convert("Asia/Tokyo")
    if saved is not None:
        df = pd.concat([saved, df])
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    
    path.parent.mkdir(exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            df.to_csv(stream, index_label="Datetime")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    
    return df

def get_all_timestamps(all_data):
    # すべての銘柄をまとめたデータ（辞書型）を用いてすべてのタイムスタンプを取得する
    all_times = []
    
    for df in all_data.values():
        all_times.extend(df.index)
    
    all_times = sorted(set(all_times))
    
    return all_times


def order_reason(portfolio, order, sold_today, *, latest_prices=None, reserve_ratio=0, max_position_ratio=1):
    if not math.isfinite(reserve_ratio) or not 0 <= reserve_ratio < 1:
        raise ValueError("予備資金の割合は0以上1未満にしてください。")
    if not math.isfinite(max_position_ratio) or not 0 < max_position_ratio <= 1:
        raise ValueError("1銘柄の配分上限は0より大きく1以下にしてください。")
    symbol = order["symbol"]
    quantity = order["quantity"]
    price = order["price"]
    cost = order["cost"]

    position = portfolio["positions"].get(symbol, {})
    holding_quantity = position.get("quantity", 0)
    cash = portfolio["cash"]
    if order["action"] not in {"buy", "sell"}:
        return "keep"
    if (not math.isfinite(price) or price <= 0 or not math.isfinite(cost) or cost < 0
            or isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0):
        raise ValueError("価格・注文数量は正、コストは0以上の有限値にしてください。")

    if order["action"] == "buy":
        if symbol in sold_today:
            return "sold_today"
        required_cash = price * quantity + cost
        if cash < required_cash:
            return "insufficient_cash"
        if reserve_ratio or max_position_ratio < 1:
            assets = cash
            for code, held in portfolio["positions"].items():
                mark = (latest_prices or {}).get(code)
                if mark is None or not math.isfinite(mark) or mark <= 0:
                    raise ValueError(f"{code} の判断時点の評価価格がありません。")
                assets += held["quantity"] * mark
            assets -= cost
            if cash - required_cash < assets * reserve_ratio:
                return "reserve_limit"
            if (holding_quantity + quantity) * price > assets * max_position_ratio:
                return "position_limit"
        return None
    if quantity > holding_quantity:
        return "insufficient_holding"
    if cash + price * quantity - cost < 0:
        return "insufficient_cash"
    return None


def validate_order(portfolio, order, sold_today, **limits):
    return order_reason(portfolio, order, sold_today, **limits) is None

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
        ].dropna(subset=["Close"])

        if day_data.empty:
            raise ValueError(
                f"{symbol} の {current_time.date()} の評価価格がありません"
            )

        # 当日中に確定した直近の有効な終値で評価し、実際の価格時刻も記録する。
        last_row = day_data.iloc[-1]

        latest_prices[symbol] = {
            "latest_price": float(last_row["Close"]),
            "price_datetime": last_row.name,
        }

    return latest_prices, valuation_time
