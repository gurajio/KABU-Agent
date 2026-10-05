# KABU-Agent

## 保存済みデータからシミュレーションする

WindowsのPowerShellでKABU-Agentフォルダを開きます。現在は `data/` に保存済みの
49銘柄と日経平均を使うため、株価の再取得は不要です。

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe simulation.py --strategy rf --period validation
.\.venv\Scripts\python.exe simulation.py --strategy rf --period test
```

- 学習：2026年8月7日〜9月7日。検証：9月8日〜9月17日。テスト：9月18日〜10月2日。
- 新しいモデルは `models/rf_current.pkl`。旧モデルは残します。既に存在する場合は
  学習を繰り返さずシミュレーションへ進むか、`train.py --model models/別名.pkl` で保存し、
  `simulation.py --model models/別名.pkl` で同じモデルを指定します。
- CSVが変わった場合は再学習します。モデルと入力CSVのハッシュ・学習条件・
  scikit-learnバージョンが一致しない場合は停止します。
- 欠損は補完せず、その足や履歴不足の特徴量ではRF判断を行いません。
  日またぎ・昼休み・足の欠落・欠損で特徴量の履歴を区切ります。
- 日末の評価は同日中の直近の有効な終値を使い、価格の時刻も記録します。
  当日の有効な終値がない場合は停止します。
- 初期資金100万円、注文単位100株、売買閾値0.55/0.45、手数料・スリッページ0は
  既存シミュレーションの設定です。業種・保有比率の制限は未実装です。
- 結果は `logs/rf_<期間>_<実行日時>/` に保存します。売買、資産推移、最終保有、
  予測確率、実行条件を記録し、以前の結果は上書きしません。
- `main.py` は最新株価の取得用の既存処理です。過去データのシミュレーションは
  `simulation.py` を使用してください。

新しいPCの環境準備：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

テストは外部通信を使いません。

```powershell
.\.venv\Scripts\python.exe -m unittest test_setup_data test_simulation
```

## 日単位の資産推移を画像にする

`visualize_logs.py` 上部の `LOG_FOLDER` に、`logs/` 内のフォルダ名を書きます。

```python
LOG_FOLDER = "random_validation_20261005_144505_247557"
```

```powershell
.\.venv\Scripts\python.exe visualize_logs.py
```

指定フォルダの `trade_logs.json` と `equity_logs.json` から、総資産・現金・保有株評価額と
日別の買い・売り件数を描き、同じフォルダに `visualization_<実行日時>.png` を保存します。
同日に複数の資産記録がある場合は最後の記録を使用します。記録のない日の資産は補完しません。
必要なら実行時の引数で別のフォルダを指定することもできます。
フォルダは `logs/フォルダ名` や絶対パスでも指定できます。
ウィンドウ表示も必要なら `--show` を付けます。既存の画像は上書きしません。

## 初回の株価データ取得（Windows）

`setup_data.py` は `tickers.csv` の全49銘柄と日経平均 `^N225` を順番に取得します。
5分足は直近60日以内に限られるため、既定では実行時点で取得可能な最古の
丸一日から2026年10月2日までを指定します。10月5日の実行なら開始日は8月7日です。
8月5日開始を明示した場合は範囲外として通信前に停止します。
学習開始日も8月7日に設定しています。

プロジェクトのフォルダで、まず条件を確認します。

```powershell
.\.venv\Scripts\python.exe setup_data.py
```

条件を確認した後、取得を実行します。

```powershell
.\.venv\Scripts\python.exe setup_data.py --download
```

- 保存先：`data/<symbol>_5m.csv`。日時列は `Datetime`（Asia/Tokyo、足の開始時刻）、
  価格列は `Open, High, Low, Close, Volume`、日時の昇順。価格調整は `auto_adjust=False`。
- 同名のJSONに取得条件・取得日時・件数・範囲・ハッシュを記録します。
  条件と内容が一致する正常なCSVは再読み込みして再利用します。
  記録のない既存CSVや条件の異なるCSVは上書きせず、その銘柄を失敗として報告します。
- 取得結果は `data/setup_<実行日時>.json` に保存し、標準出力には銘柄別結果と件数を表示します。
- 銘柄間は2秒、制限エラー時は60秒・120秒の待機後に最大2回再試行します。
  制限が続いた場合は残りの取得も停止します。制限以外のエラーは再試行せず次へ進みます。
  間隔は `--interval`、制限時の待機は `--cooldown`、再試行上限は `--retries` で変更できます。
  これは設定値であり、制限エラーを防ぐ保証ではありません。
- 50件の通常待機は約1.6分に通信時間を加えた長さです。制限時は長くなります。
- OHLCVの値と日時は検証しますが、休場・無約定・配信欠損の判別と期間全体の
  網羅性は未確認です。保存成功は学習期間全体のデータが揃ったことを保証しません。
- 別条件の取得は `--start YYYY-MM-DD --end YYYY-MM-DD --output data_other`
  で別フォルダへ保存できます。終了日は含めます。

`data/` はGit管理の対象外です。旧パソコンの保存データは別途移してください。
