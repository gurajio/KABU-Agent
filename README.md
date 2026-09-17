# KABU-Agent

銘柄一覧CSVを読み込み、yfinanceから5分足を取得して、PandasのDataFrameを標準出力します。

- `main.py`：CSV読み込み・株価取得・表示
- `tickers.csv`：銘柄コード（`symbol`）と会社名（`name`）の一覧

KABU-Agentフォルダ内で実行します。

```sh
.venv/bin/python main.py
```

銘柄を変更する場合は `tickers.csv` を編集してください。`symbol` には `9432.T` のようなコードを入れます。
5分足の期間は `period="max"` で取得可能な最大期間を指定しています。
株価データは保存せず、全行を表示します。

別の環境で準備する場合：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install pandas yfinance
```
