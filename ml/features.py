from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

BAR_SIZE = pd.Timedelta(minutes=5)
FEATURE_COLUMNS = (
    "range_ratio", "close_position", "return_mean", "return_std",
    "return_autocorr", "adx", "relative_volume", "price_volume_corr",
    "market_return", "elapsed_minutes",
)


@dataclass(frozen=True)
# 計算条件をまとめる（文字列の種類を決定）
class FeatureConfig:
    mean_window: int
    std_window: int
    autocorr_window: int
    adx_window: int
    adx_warmup: int
    volume_window: int
    corr_window: int
    timestamp_kind: str
    sessions: tuple[tuple[str, str], ...]
    market_name: str
    # 設定作成後に自動で実行される関数
    def __post_init__(self):
        # 数値の設定を順番に確認
        for name, minimum in (
            # (設定項目の名前,許容する最小値)
            ("mean_window", 1), ("std_window", 2), ("autocorr_window", 3),
            ("adx_window", 2), ("adx_warmup", 0), ("volume_window", 1),
            ("corr_window", 3),
        ):
            # getattrとは、文字列で指定した名前の属性を取り出す関数
            _check_integer(getattr(self, name), name, minimum)
        if self.timestamp_kind not in {"start", "end"}:
            raise ValueError("timestamp_kind は start または end を指定してください。")
        if not isinstance(self.market_name, str) or not self.market_name.strip():
            raise ValueError("使用する指数名 market_name が必要です。")
        sessions = tuple(tuple(pair) for pair in self.sessions)
        if not sessions or any(len(pair) != 2 for pair in sessions):
            raise ValueError("sessions に取引開始・終了時刻の組を指定してください。")
        previous = -1
        for start, end in sessions:
            start, end = _clock_minutes(start), _clock_minutes(end)
            if start < previous or start >= end or (end - start) % 5:
                raise ValueError("取引時間は昇順・重複なし・5分単位で指定してください。")
            previous = end
        object.__setattr__(self, "sessions", sessions)


def _check_integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} は {minimum} 以上の整数が必要です。")


def _clock_minutes(value):
    parsed = time.fromisoformat(value)
    if parsed.second or parsed.microsecond or parsed.tzinfo:
        raise ValueError("取引時間は HH:MM 形式で指定してください。")
    return parsed.hour * 60 + parsed.minute


def _as_timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("判断時刻にはタイムゾーンを指定してください。")
    return stamp.tz_convert("Asia/Tokyo")


def _prepare_frame(data, config, *, market=False, as_of=None):
    columns = ["Close"] if market else ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("入力にはタイムゾーン付き DatetimeIndex が必要です。")
    if data.index.hasnans or data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("入力日時は欠損・重複なしの昇順にしてください。")
    if data.columns.has_duplicates or not set(columns).issubset(data.columns):
        raise ValueError(f"重複のない必要列 {columns} を用意してください。")
    index = data.index.tz_convert("Asia/Tokyo")
    if config.timestamp_kind == "start":
        index = index + BAR_SIZE
    frame = data.loc[:, columns].copy()
    frame.index = index.rename("decision_time")
    if as_of is not None:
        frame = frame.loc[frame.index <= _as_timestamp(as_of)]
    frame = frame.astype(float)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("入力に無限大が含まれています。")
    prices = frame.drop(columns="Volume", errors="ignore")
    if (prices <= 0).any().any():
        raise ValueError("価格は正の値が必要です。")
    if not market:
        invalid = (
            (frame.Volume < 0) | (frame.High < frame.Low)
            | (frame.High < frame[["Open", "Close"]].max(axis=1))
            | (frame.Low > frame[["Open", "Close"]].min(axis=1))
        )
        if invalid.any():
            raise ValueError("OHLCの大小関係または出来高が不正です。")
    return frame


def _segments(frame, config):
    index = frame.index
    minutes = (index - index.normalize()).total_seconds() / 60
    session = pd.Series(-1, index=index, dtype=int)
    for number, (start, end) in enumerate(config.sessions):
        start, end = _clock_minutes(start), _clock_minutes(end)
        inside = (minutes >= start + 5) & (minutes <= end) & ((minutes - start) % 5 == 0)
        session.loc[inside] = number
    valid = frame.notna().all(axis=1) & session.ge(0)
    times = index.to_series()
    breaks = (
        session.ne(session.shift()) | times.diff().ne(BAR_SIZE)
        | times.dt.normalize().ne(times.dt.normalize().shift())
        | ~valid | ~valid.shift(fill_value=False)
    )
    return breaks.cumsum().where(valid)


def _correlation(left, right):
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return np.nan
    return float(np.clip(np.dot(left, right) / denominator, -1, 1))


def _lag_corr(values):
    return _correlation(values[:-1], values[1:])


# 初期値をn個の単純平均とし、未定義値は次の区切りまで引き継ぐ。
def _wilder(values, window, start):
    result = np.full(len(values), np.nan)
    seed = start + window - 1
    if seed < len(values):
        result[seed] = np.mean(values[start:seed + 1])
        for position in range(seed + 1, len(values)):
            result[position] = ((window - 1) * result[position - 1] + values[position]) / window
    return result


def _adx(frame, config):
    upward = frame.High.diff()
    downward = -frame.Low.diff()
    positive = upward.where((upward > downward) & (upward > 0), 0)
    negative = downward.where((downward > upward) & (downward > 0), 0)
    ranges = pd.concat([
        frame.High - frame.Low,
        (frame.High - frame.Close.shift()).abs(),
        (frame.Low - frame.Close.shift()).abs(),
    ], axis=1).max(axis=1)
    n = config.adx_window
    smooth = [_wilder(values.to_numpy(), n, 1) for values in (positive, negative, ranges)]
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = 100 * smooth[0] / smooth[2]
        minus = 100 * smooth[1] / smooth[2]
        dx = 100 * np.abs(plus - minus) / (plus + minus)
    adx = _wilder(dx, n, n)
    adx[:2 * n - 1 + config.adx_warmup] = np.nan
    return adx


# 入出力は1銘柄分。指数も同じ時刻表記を使い、出力のindexは足の確定時刻。
def build_features(prices, market, config, *, as_of=None):
    frame = _prepare_frame(prices, config, as_of=as_of)
    index_frame = _prepare_frame(market, config, market=True, as_of=as_of)
    result = pd.DataFrame(np.nan, index=frame.index, columns=FEATURE_COLUMNS)
    market_return = pd.Series(np.nan, index=index_frame.index)
    for _, part in index_frame.groupby(_segments(index_frame, config)):
        market_return.loc[part.index] = part.Close.pct_change(fill_method=None)
    for _, part in frame.groupby(_segments(frame, config)):
        features = pd.DataFrame(index=part.index)
        span = part.High - part.Low
        returns = part.Close.pct_change(fill_method=None)
        growth = part.Volume.div(part.Volume.shift().replace(0, np.nan)) - 1
        features["range_ratio"] = span / part.Open
        features["close_position"] = (part.Close - part.Low) / span.replace(0, np.nan)
        features["return_mean"] = returns.rolling(config.mean_window).mean()
        features["return_std"] = returns.rolling(config.std_window).std(ddof=1)
        features["return_autocorr"] = returns.rolling(config.autocorr_window).apply(_lag_corr, raw=True)
        features["adx"] = _adx(part, config)
        average = part.Volume.shift().rolling(config.volume_window).mean()
        features["relative_volume"] = part.Volume / average.replace(0, np.nan)
        paired = pd.concat([returns, growth], axis=1).to_numpy()
        correlation = np.full(len(part), np.nan)
        for position in range(config.corr_window - 1, len(part)):
            window = paired[position - config.corr_window + 1:position + 1]
            if np.isfinite(window).all():
                correlation[position] = _correlation(window[:, 0], window[:, 1])
        features["price_volume_corr"] = correlation
        features["market_return"] = market_return.reindex(part.index)
        minutes = (part.index - part.index.normalize()).total_seconds() / 60
        features["elapsed_minutes"] = minutes - _clock_minutes(config.sessions[0][0])
        result.loc[part.index] = features.loc[:, FEATURE_COLUMNS]
    return result.replace([np.inf, -np.inf], np.nan)
