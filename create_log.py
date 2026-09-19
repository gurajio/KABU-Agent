import json
from pathlib import Path

def create_trade_log(order, portfolio, current_time):
    return {
        "datetime": str(current_time),
        "symbol": order["symbol"],
        "action": order["action"],
        "quantity": order["quantity"],
        "price": order["price"],
        "cost": order["cost"],
        "cash_after": portfolio["cash"],
    }

def create_equity_log(portfolio, latest_prices, valuation_time):
    # 評価に使う価格は、銘柄ごとに latest_price と price_datetime を渡す
    positions = {}
    cash = float(portfolio["cash"])
    total_assets = cash

    for symbol, position in portfolio["positions"].items():
        quantity = int(position["quantity"])
        price_data = latest_prices[symbol]
        price = float(price_data["latest_price"])

        # 後からportfolioが変わってもログが変わらないよう、新しい辞書を作る
        positions[symbol] = {
            "quantity": quantity,
            "latest_price": price,
            "price_datetime": str(price_data["price_datetime"]),
        }
        total_assets += quantity * price

    return {
        "datetime": str(valuation_time),
        "positions": positions,
        "cash": cash,
        "total_assets": total_assets,
    }


def save_logs(trade_logs, equity_logs, output_dir="logs"):
    # 1回のシミュレーションの全ログを、1つのJSONファイルにまとめる
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # logs = {
    #     "trades": trade_logs,
    #     "equity": equity_logs,
    # }

    # with open(path / "simulation.json", "w", encoding="utf-8") as file:
    #     json.dump(logs, file, ensure_ascii=False, indent=4)
    
    with open(path / "trade_logs.json", "w", encoding="utf-8") as file:
        json.dump(trade_logs, file, ensure_ascii=False, indent=4)
    
    with open(path / "equity_logs.json", "w", encoding="utf-8") as file:
        json.dump(equity_logs, file, ensure_ascii=False, indent=4)
