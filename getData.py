import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
BAR_SIZE = pd.Timedelta(minutes=5)


def validate_data(frame, start, end, *, symbol=""):
    if frame.empty or not set(COLUMNS).issubset(frame.columns):
        raise ValueError("必要なOHLCVデータを取得できませんでした。")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("タイムゾーン付きの日時が必要です。")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("取得日時に欠損・重複・順序の異常があります。")
    frame = frame.loc[:, COLUMNS].copy()
    frame.index = index.tz_convert("Asia/Tokyo").rename("Datetime")
    if (frame.index < start).any() or (frame.index >= end).any():
        raise ValueError("指定範囲外のデータが返されました。")
    if (frame.index != frame.index.floor("5min")).any():
        raise ValueError("5分足の開始時刻と一致しない日時があります。")
    frame = frame.loc[frame.index + BAR_SIZE <= end]
    if symbol == "^N225" or symbol.endswith(".T"):
        minutes = frame.index.hour * 60 + frame.index.minute
        lunch = (minutes >= 11 * 60 + 30) & (minutes < 12 * 60 + 30)
        empty = frame[COLUMNS[:4]].isna().all(axis=1) & frame.Volume.eq(0)
        excluded = lunch & empty
        if excluded.any():
            print(f"昼休みの空行を除外: {int(excluded.sum())}行（OHLC全欠損・出来高0）")
            frame = frame.loc[~excluded]
    if frame.empty:
        raise ValueError("指定期間に確定済みの5分足がありません。")
    frame = frame.astype(float)
    missing = ~np.isfinite(frame)
    if missing.any().any():
        counts = missing.sum()
        first = frame.index[missing.any(axis=1)][0]
        raise ValueError(f"欠損または非有限値: {counts[counts > 0].to_dict()}、最初の時刻: {first}")
    invalid = (
        (frame[COLUMNS[:4]] <= 0).any(axis=1) | (frame.Volume < 0)
        | (frame.High < frame.Low)
        | (frame.High < frame[["Open", "Close"]].max(axis=1))
        | (frame.Low > frame[["Open", "Close"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError("取得データの価格・出来高・OHLCの大小関係が不正です。")
    return frame


# 書き込みを終えた一時ファイルから保存先を作り、既存ファイルは上書きしない。
def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index_label="Datetime")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="指定銘柄・指数の確定済み5分足を取得します。")
    parser.add_argument("symbol", help="例: 9432.T または ^N225")
    parser.add_argument("--days", type=int, required=True, help="取得する直近の暦日数（1〜60）")
    parser.add_argument("--output", type=Path, help="getData.pyの場所を基準とする保存先。絶対パスも指定可能")
    args = parser.parse_args()
    if not 1 <= args.days <= 60:
        parser.error("--daysは1〜60を指定してください。")
    if not args.symbol.strip() or any(char in args.symbol for char in '/\\'):
        parser.error("銘柄コードに空文字やパス区切りは指定できません。")
    path = (ROOT / (args.output or Path("data") / f"{args.symbol}_5m.csv")).resolve()
    if path.exists():
        parser.error(f"保存先が存在します。--outputで別名を指定してください: {path}")
    end = pd.Timestamp.now(tz="Asia/Tokyo")
    start = (end - pd.Timedelta(days=args.days)).ceil("5min")
    print(f"取得条件: {args.symbol}, 5分足, auto_adjust=False, {start} 以上〜{end} 未満")
    hide_errors = yf.config.debug.hide_exceptions
    yf.config.debug.hide_exceptions = False
    try:
        frame = yf.Ticker(args.symbol).history(
            start=start, end=end, interval="5m", auto_adjust=False,
            actions=False, keepna=True,
        )
    finally:
        yf.config.debug.hide_exceptions = hide_errors
    frame = validate_data(frame, start, end, symbol=args.symbol)
    save_csv(frame, path)
    print(f"保存完了: {len(frame)}行、足の開始時刻 {frame.index[0]} 〜 {frame.index[-1]}")
    print(f"保存先: {path}")


if __name__ == "__main__":
    main()
