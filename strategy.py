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
    # 仮に実装しているだけなので、取引数や手数料等は省いている
    # 対象銘柄の保有情報を取得する
    position = portfolio["positions"].get(symbol_code, {})
    has_position = position.get("quantity", 0) > 0

    value = rng.random()
    
    # 株価を取得
    stock_price = symbol_dataframe.loc[datetime]["Open"]
    quantity = 100
    cost = 0
    
    if value < buy_probability:
        action = "buy"

    elif has_position and value < buy_probability + sell_probability:
        action = "sell"

    else:
        action = "keep"

    return {
            "symbol": symbol_code,
            "price": stock_price,
            "quantity":quantity,
            "action": action,
            "reason": f"ランダム条件で{action}と判断",
            "cost":cost,
        }


def rf_strategy(symbol_code, price, portfolio, probability, *, buy_threshold=0.40, sell_threshold=0.35):
    if not 0 <= sell_threshold < buy_threshold <= 1:
        raise ValueError("閾値は 0 ≦ 売り < 買い ≦ 1 にしてください。")
    holding = portfolio["positions"].get(symbol_code, {}).get("quantity", 0)
    action = "keep"
    if pd.isna(probability):
        reason = "上昇確率を計算できないため売買なし"
    else:
        if not 0 <= probability <= 1:
            raise ValueError("上昇確率は0〜1にしてください。")
        if probability >= buy_threshold:
            action = "buy"
        elif probability <= sell_threshold and holding >= 100:
            action = "sell"
        reason = f"上昇確率={probability:.6f}により{action}"
    return {
        "symbol": symbol_code, "price": float(price), "quantity": 100,
        "action": action, "reason": reason, "cost": 0,
    }
