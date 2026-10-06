# KABU-Agent

日本株の保存済み5分足を使い、ランダムフォレストによる予測と仮想売買を行います。
実際の証券口座への発注は行いません。

## 保存済みデータからシミュレーションする

WindowsのPowerShellでKABU-Agentフォルダを開きます。現在は `data/` に保存済みの
49銘柄と日経平均を使うため、株価の再取得は不要です。

### 1. 新12特徴量のモデルを学習する

通常の学習ではモデルを1つ作成します。`--compare` は候補を比較したい場合に使う任意の操作です。
新12特徴量（`momentum12`）が既定値なので、特徴量の指定は省略できます。

```powershell
.\.venv\Scripts\python.exe train.py --model models/rf_momentum12.pkl
```

保存先に同名のモデルがある場合は、上書きせず停止します。再学習するときは
`models/rf_momentum12_v2.pkl` など別名を指定し、以下のシミュレーションでも同じファイルを指定してください。

### 2. 検証データでシミュレーションする

```powershell
.\.venv\Scripts\python.exe simulation.py --strategy rf --period validation --model models/rf_momentum12.pkl
```

初期資金100万円、買い閾値0.42・売り閾値0.35で検証する例：

```powershell
.\.venv\Scripts\python.exe simulation.py --strategy rf --period validation --model models/rf_momentum12.pkl --initial-cash 1000000 --buy-threshold 0.42 --sell-threshold 0.35
```

閾値は `simulation.py` の引数で指定します。`strategy.py` の関数の既定値を変更しても、
シミュレーションから渡される値が優先されます。

RFで検証期間を実行すると、評価件数・正解件数・正答率と、多数派を常に予測した場合の
基準正答率も表示します。確率0.5以上を上昇、未満を非上昇とし、判断時刻のOpenから
10分後のOpenへの変化と照合します。同値は非上昇です。欠損や昼休み・引けなどで
正解を作れない予測は除外します。売買閾値・約定の有無は正答率の計算に影響しません。
結果は `settings.json` の `validation_metrics` に保存します。`accuracy` は0〜1の比率です。

### 3. 設定を決めてからテストデータで評価する

```powershell
.\.venv\Scripts\python.exe simulation.py --strategy rf --period test --model models/rf_momentum12.pkl
```

検証時に資金や閾値を変更した場合は、テスト時にも同じ引数を付けます。
モデル・特徴量・閾値は検証期間で選び、テスト結果を見て調整し直さないようにします。

| 用途 | 期間 |
|---|---|
| 学習 | 2026年8月7日〜9月7日 |
| 検証・設定選択 | 2026年9月8日〜9月17日 |
| テスト・最終評価 | 2026年9月18日〜10月2日 |

### 既存モデルを使う場合

```powershell
.\.venv\Scripts\python.exe simulation.py --strategy rf --period validation
```

`--model` を省略すると `models/rf_current.pkl` を使います。既存の10特徴量モデルは
その構成のまま予測するため、新12特徴量を反映するには新しく学習したモデルを指定します。
`train.py` だけで実行すると保存先も `models/rf_current.pkl` になり、既に存在する場合は停止します。

### 実行条件と保存結果

- CSVが変わった場合は再学習します。モデルと入力CSVのハッシュ・学習条件・
  scikit-learnバージョンが一致しない場合は停止します。
- 欠損は補完せず、その足や履歴不足の特徴量ではRF判断を行いません。
  日またぎ・昼休み・足の欠落・欠損で特徴量の履歴を区切ります。
- 日末の評価は同日中の直近の有効な終値を使い、価格の時刻も記録します。
  当日の有効な終値がない場合は停止します。
- 初期資金の既定値は `simulation.py` の `INITIAL_CASH`（現在1億円）、注文単位は100株、売買閾値は
  0.55/0.45、手数料・スリッページは0です。初期資金は `--initial-cash` で指定できます。
- 購入後に総資産の10%を現金で残し、1銘柄の保有額は買い増しを含め総資産の10%以下に
  制限します。判定には判断時点のOpen、足がなければ過去の確定Closeを使い、将来の値は
  使いません。価格変動による上限超過での強制売却・業種別の制限は未実装です。
  `--cash-reserve` と `--position-limit` は0〜1の比率で指定します。
- 結果は `logs/rf_<期間>_<実行日時>/` に保存します。売買、資産推移、最終保有、
  予測確率、実行条件を記録し、以前の結果は上書きしません。
  `settings.json` の `order_counts` で資金不足・予備資金制限・銘柄上限などの見送り数を
  確認できます。売買ログには取引直後の現金比率と対象銘柄の保有比率も記録します。
- `main.py` は最新株価の取得用の既存処理です。過去データのシミュレーションは
  `simulation.py` を使用してください。

新しいPCの環境準備：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

テストは外部通信を使いません。

```powershell
.\.venv\Scripts\python.exe -m unittest test_setup_data test_simulation test_train
```

## 入力特徴量

新しい学習の既定値は `momentum12`（12特徴量）です。

| 分類 | 特徴量 | 列数 |
|---|---|---:|
| ローソク足 | 相対値幅、終値位置、符号付き実体 | 3 |
| 値動き | 5・15・30分収益率 | 3 |
| 変動性 | 直近6収益率の標本標準偏差（ddof=1） | 1 |
| トレンド | ADX（期間5） | 1 |
| 出来高 | 相対出来高（最新出来高÷直前6本の平均） | 1 |
| 市場との関係 | 日経平均の5分収益率、銘柄と日経平均の15分収益率の差 | 2 |
| 時間 | 寄り付きからの経過分数（昼休みを含む） | 1 |
| 合計 | | 12 |

収益率は確定足の終値から計算します。日またぎ・昼休み・欠損をまたがず、分母0や履歴不足は
NaNとして判断対象から除外します。財務指標・業種比較は含みません。

予測対象は、判断時刻のOpenから10分後のOpenまでの上昇・非上昇です。
10分後に必ず売却するルールではありません。

## モデル候補を比較する（任意）

学習の進み具合を見る場合は、次のコマンドで各候補の木の本数を段階的に増やした時点の
学習正答率・検証正答率・検証正解件数を標準出力します。RFにはepochがないため、
既存の木を残して追加する段階を表示します。検証データは正答率の計算だけに使います。

```powershell
.\.venv\Scripts\python.exe -u train.py --compare --feature-set momentum12 --learning-curve
```

各候補の履歴は、新しい比較結果フォルダの `<候補名>_learning_curve.csv` に保存します。
正答率は確率0.5で判定し、CSVにはAUC・Brier・正解件数・評価件数・多数派の基準も記録します。
最終的な木の本数は `FOREST_PARAMS["n_estimators"]` です。例えば200なら
10・20・50・100・200本で評価します。最終本数より大きい途中段階は省略します。
保存される各候補のモデルは最終本数のモデルです。途中の最良モデルは自動採用しません。

旧10特徴量との比較は `train.py --compare --feature-set legacy10`、新12特徴量は
`train.py --compare --feature-set momentum12` で別保存できます。対象となる行数が異なるため、
両構成の性能を直接比べる場合は予測CSVの symbol・decision_time の共通行で評価してください。
既存の10特徴量モデルはその構成で引き続き予測します。12特徴量を使うには新しいモデルを
学習し、シミュレーションの `--model` にそのファイルを指定します。自動採用は行いません。

現在の木の深さ5と、深さ10・20・制限なしの候補を検証期間で比較します。
葉の最低標本数も1・20・100の組み合わせを含む計6候補です。
学習期間だけで学習し、検証期間だけで精度を計算します。テスト期間は使いません。

```powershell
.\.venv\Scripts\python.exe train.py --compare --feature-set momentum12
```

結果は新規の `logs/model_validation_<実行日時>/` に保存します。
`comparisons.csv` に検証AUC・Brier・log loss、学習AUC、確率の平均・標準偏差・分位点を
記録し、各候補のモデルと正解がそろう検証行の予測CSV、実行条件JSONを併せて保存します。
AUCは大きいほど、Brier・log lossは小さいほど良い指標です。定数予測とのBrierの比較と、
学習・検証AUCの差も確認します。確率が広がったことだけを改善とは扱いません。
`models/rf_current.pkl` の採用モデルは変更しません。比較途中の失敗はJSONに記録します。

候補を選んだ後は、保存された `.pkl` をシミュレーションの `--model` で指定します。
例えば、実際の結果フォルダ名を使って次のように実行します。

```powershell
.\.venv\Scripts\python.exe simulation.py --strategy rf --period validation --model "logs/model_validation_<実行日時>/depth_5_leaf_1.pkl"
```

`<実行日時>` は、比較実行時に表示された保存先の日時へ置き換えてください。

学習設定は `train.py` 上部の `FOREST_PARAMS` を編集します。通常学習・銘柄別学習では
辞書の値をそのまま使い、実行時に実際の設定を表示します。`--max-depth`・
`--min-samples-leaf` は互換用に受け付けますが、指定値は使わずコード内設定を優先します。
`--compare` の深さ・葉数はコード内の `MODEL_CANDIDATES` を使い、他の設定は
`FOREST_PARAMS` を使います。検証で設定を決めてから、テスト期間で最終評価します。

## 共通モデルと銘柄別モデルを比較する

`FOREST_PARAMS` に指定した同じ学習設定で、49銘柄の共通モデル1個と
銘柄別モデル49個を学習します。個別モデルは順次学習し、各モデル内部の使用コア数を表示します。
テスト期間は使いません。木の本数・深さ・葉の最低標本数などは `FOREST_PARAMS` を編集します。

```powershell
.\.venv\Scripts\python.exe -u train.py --per-symbol --feature-set momentum12
```

保存先は新しい `logs/symbol_validation_<実行日時>/` です。`--output` で別の新規フォルダも指定できます。

銘柄ごとの正答率に加え、最後に共通モデル・個別モデルそれぞれの全体検証正答率と
正解件数・評価件数を表示します。銘柄別の率の単純平均ではなく、正解件数を合算して
評価件数の合計で割ります。比較が完了した同じ銘柄・時刻だけを使い、見送り銘柄は除外します。
集計は `settings.json` の `overall_validation` に保存します。

| ファイル | 内容 |
|---|---|
| `common.pkl` | 全銘柄をまとめて学習した比較基準 |
| `<銘柄>.pkl` | その銘柄だけで学習したモデル |
| `symbol_comparisons.csv` | 銘柄別の共通・個別正答率、差、AUC、Brier、評価件数、基準正答率 |
| `<銘柄>_predictions.csv` | 同じ検証行の正解と共通・個別の予測確率 |
| `selection.json` | 全銘柄を共通モデルに対応させた選択用の初期ファイル |
| `individual_models.json` | 個別モデルを試すための対応表。学習不能の銘柄は共通モデル |
| `settings.json` | 学習条件、成功・見送り件数、実行状態 |

個別モデルを採用したい銘柄だけ、`selection.json` の値を `common.pkl` から
`9432.T.pkl` のように変更します。モデルのパスはJSONファイルのあるフォルダからの相対パスです。
自動採用は行いません。1クラスしかない等で学習できなかった銘柄は、理由をCSVに記録します。
検証に1クラスしかない場合は正答率を計算し、AUCは空欄にします。

```powershell
.\.venv\Scripts\python.exe simulation.py --strategy rf --period validation --model "logs/symbol_validation_<実行日時>/selection.json" --initial-cash 1000000 --buy-threshold 0.45 --sell-threshold 0.42 --cash-reserve 0.10 --position-limit 0.10
```

`<実行日時>` を実際の保存先の日時に置き換えてください。全個別モデルで試す場合は
`selection.json` を `individual_models.json` に変更します。選択を固定した後のテスト実行では
`--period test` に変更します。1口座の資金を全銘柄で共有し、既存の購入制限を適用します。
対応表は対象全銘柄を含む必要があり、モデル間の特徴量・学習期間・CSVの不一致は停止します。

比較は現在の1検証期間です。複数期間での安定性はまだ確認していません。
既に確認したテスト期間の再実行は参考評価として扱います。

## 特徴量の相関と寄与を確認する

保存済みモデルについて、検証期間だけで特徴量のPearson・Spearman相関と
Permutation Importanceを計算します。学習済みモデルの変更・再学習は行いません。

```powershell
.\.venv\Scripts\python.exe -u analyze_features.py --model logs/model_validation_20261006_104446_232336/depth_5_leaf_1.pkl
```

結果は新しい `logs/feature_analysis_<実行日時>/` に保存します。`correlation.png` は
Spearman相関の図、`correlation_pairs.csv` は相関の強い順の一覧、
`permutation_importance.csv` は入力1列をシャッフルしたときの性能低下です。
`auc_drop_mean`・`accuracy_drop_mean` が正なら元の性能が高く、`brier_drop_mean` が
正ならシャッフルで誤差が増加しました。いずれも大きいほど現在のモデルがその列に
依存しています。`accuracy_drop_mean` は0〜1の比率です。
反復は既定5回で、`--repeats` で変更できます。

相関0は独立を保証せず、相関のある特徴量は寄与が低く見える場合があります。
標準偏差はシャッフルによる変動で、未知期間での信頼区間ではありません。
指標の削除は、同じ評価行で再学習したモデルを比較してから判断します。

## 予測確率の度数分布を画像にする

`visualize_predictions.py` で予測CSVの確率を5パーセントポイント刻みのグラフにします。
引数を省略すると、スクリプト上部の `PREDICTIONS` に指定したCSVを使います。

```powershell
.\.venv\Scripts\python.exe visualize_predictions.py --predictions logs/rf_validation_20261006_143552_826902/predictions.csv
```

画像と度数分布CSVは入力CSVと同じフォルダに日時付きで保存します。欠損は除外し、
不正な確率や有効予測がない場合は停止します。`--bin-width 1` で1ポイント刻み、
`--show` でウィンドウ表示もできます。区間は下限以上・上限未満で、最後の区間は100%も含みます。

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
