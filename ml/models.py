import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from .features import (
    BAR_SIZE, FEATURE_COLUMNS, FeatureConfig, _as_timestamp, _check_integer,
    _prepare_frame, _segments, build_features,
)


@dataclass
class ModelBundle:
    classifier: RandomForestClassifier
    config: FeatureConfig
    horizon: int
    train_end: pd.Timestamp
    train_rows: int
    symbols: tuple[str, ...]
    sklearn_version: str
    feature_names: tuple[str, ...] = FEATURE_COLUMNS
    schema_version: int = 1
    data_hash: str | None = None
    train_start: str | None = None


# 確定足の次のOpenからhorizon本後のOpenへの変化を正解とする。時刻列は約定想定時刻。
def build_labels(prices, config, *, horizon, as_of=None):
    _check_integer(horizon, "horizon", 1)
    frame = _prepare_frame(prices, config, as_of=as_of)
    result = pd.DataFrame(np.nan, index=frame.index, columns=["target", "future_return"])
    for name in ("entry_time", "exit_time"):
        result[name] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, Asia/Tokyo]")
    for _, part in frame.groupby(_segments(frame, config)):
        entry = part.Open.shift(-1)
        exit_price = part.Open.shift(-(horizon + 1))
        change = exit_price / entry - 1
        target = change.gt(0).astype(float).where(change.notna())
        times = part.index.to_series() - BAR_SIZE
        result.loc[part.index, "target"] = target
        result.loc[part.index, "future_return"] = change
        result.loc[part.index, "entry_time"] = times.shift(-1).where(change.notna())
        result.loc[part.index, "exit_time"] = times.shift(-(horizon + 1)).where(change.notna())
    return result


# pricesは{銘柄名: OHLCV}。train_endまでに確定した足だけで特徴量・正解を作る。
def fit_model(prices, market, config, *, horizon, train_end, forest_params, workers):
    _check_integer(horizon, "horizon", 1)
    _check_integer(workers, "workers", 1)
    limit = max(1, (os.cpu_count() or 1) - 1)
    if workers > limit:
        raise ValueError(f"workers は {limit} 以下にしてください。")
    if "n_jobs" in forest_params:
        raise ValueError("並列数は forest_params ではなく workers で指定してください。")
    for name in ("n_estimators", "random_state"):
        if name not in forest_params:
            raise ValueError(f"forest_params に {name} を明示してください。")
    _check_integer(forest_params["n_estimators"], "n_estimators", 1)
    _check_integer(forest_params["random_state"], "random_state", 0)
    if not prices or any(not isinstance(symbol, str) or not symbol for symbol in prices):
        raise ValueError("prices は銘柄名をキーとする空でない辞書にしてください。")
    cutoff = _as_timestamp(train_end)
    rows, targets = [], []
    for symbol in sorted(prices):
        features = build_features(prices[symbol], market, config, as_of=cutoff)
        labels = build_labels(prices[symbol], config, horizon=horizon, as_of=cutoff)
        valid = features.notna().all(axis=1) & labels.target.notna() & labels.exit_time.le(cutoff)
        rows.append(features.loc[valid])
        targets.append(labels.loc[valid, "target"])
    features = pd.concat(rows)
    target = pd.concat(targets).astype(int)
    if features.empty or target.nunique() != 2:
        raise ValueError("学習には、欠損のない特徴量と上昇・非上昇の両方の正解が必要です。")
    classifier = RandomForestClassifier(**forest_params, n_jobs=workers)
    print(f"ランダムフォレスト: 使用コア数={workers}, 学習行数={len(features)}")
    with threadpool_limits(limits=1):
        classifier.fit(features, target)
    return ModelBundle(
        classifier, config, horizon, cutoff, len(features),
        tuple(sorted(prices)), sklearn.__version__,
    )


def _validate_bundle(bundle):
    if not isinstance(bundle, ModelBundle) or bundle.schema_version != 1:
        raise ValueError("対応していないモデル形式です。")
    if bundle.sklearn_version != sklearn.__version__:
        raise ValueError(f"scikit-learn {bundle.sklearn_version} で保存されたモデルです。")
    if bundle.feature_names != FEATURE_COLUMNS:
        raise ValueError("モデルの特徴量が現在の実装と一致しません。")
    if tuple(bundle.classifier.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("学習済み分類器の特徴量順が一致しません。")
    if not np.array_equal(bundle.classifier.classes_, [0, 1]):
        raise ValueError("分類器にはクラス0・1の両方が必要です。")


# 学習終了後の確定足について上昇確率を返す。履歴不足・欠損・学習期間内はNaN。
def predict_up(bundle, prices, market, *, as_of):
    _validate_bundle(bundle)
    features = build_features(prices, market, bundle.config, as_of=as_of)
    result = pd.Series(np.nan, index=features.index, name="up_probability")
    valid = features.notna().all(axis=1) & (features.index > bundle.train_end)
    if valid.any():
        with threadpool_limits(limits=1):
            result.loc[valid] = bundle.classifier.predict_proba(features.loc[valid])[:, 1]
    return result


# 一時ファイルの書き込み完了後に保存先を作成し、既存ファイルは上書きしない。
def save_model(bundle, path):
    _validate_bundle(bundle)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# pickleを使うため、自分で作成した信頼できるモデルファイルだけを読み込む。
def load_model(path):
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    _validate_bundle(bundle)
    return bundle
