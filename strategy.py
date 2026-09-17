import random
import json
import pandas as pd


def random_strategy(
    datetime,
    symbol_code,
    symbol_dataframe,
    portfolio,
    rng = 43,
    buy_probability=0.1,
    sell_probability=0.1,
):
    # 対象銘柄の保有情報を取得する
    position = portfolio["positions"].get(symbol_code, {})
    has_position = position.get("quantity", 0) > 0

    value = rng.random()
    
    # 株価を取得
    stock_price = symbol_dataframe.loc[datetime]["Open"]

    if value < buy_probability:
        action = "buy"

    elif has_position and value < buy_probability + sell_probability:
        action = "sell"

    else:
        action = "keep"

    return {
            "symbol": symbol_code,
            "price": stock_price,
            "action": action,
            "reason": f"ランダム条件で{action}と判断",
        }


if __name__ == "__main__":
    # 口座状況を読み込む
    with open("portfolio.json", "r", encoding="utf-8") as file:
        portfolio = json.load(file)

    # 判断する日時
    datetime = pd.Timestamp("2026-07-21 09:05:00+09:00")

    # 株価データを読み込む
    df = pd.read_csv(
        "data/402A.T_5m.csv",
        index_col=0,
        parse_dates=[0],
    )

    # 判断時刻までに確定した足だけを取り出す
    history = df[df.index + pd.Timedelta(minutes=5) <= datetime]

    demoinputData = {
        "datetime": datetime,
        "symbol_code": "402A.T",
        "symbol_dataframe": history,
        "portfolio": portfolio,
    }

    # 乱数生成器を作る
    rng = random.Random(43)

    # 売買判断を実行して表示する
    for _ in range(10):
        result = random_strategy(**demoinputData, rng=rng)
        print(result)