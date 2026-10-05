import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from getData import ROOT, save_csv, validate_data


def write_json(value, path):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_period(start, end, now):
    end = pd.Timestamp(end, tz="Asia/Tokyo") + pd.Timedelta(days=1)
    earliest = (now - pd.Timedelta(days=60)).ceil("D")
    start = earliest if start == "earliest" else pd.Timestamp(start, tz="Asia/Tokyo")
    if start != start.normalize() or end != end.normalize():
        raise ValueError("開始日・終了日は日付だけで指定してください。")
    if start >= end or end > now:
        raise ValueError("期間の順序が不正、または終了日がまだ終了していません。")
    if start < now - pd.Timedelta(days=60):
        raise ValueError(
            f"開始日{start.date()}は5分足の直近60日制限の範囲外です。"
            f"丸一日取得できる最古の日付の目安は{earliest.date()}です。"
            "期間は自動変更しません。保存済みCSVを移すか、--startを指定してください。"
        )
    return start, end


def read_symbols(path):
    stocks = pd.read_csv(path, dtype=str)
    if "symbol" not in stocks or stocks.symbol.empty or stocks.symbol.isna().any():
        raise ValueError("tickers.csvに空でないsymbol列が必要です。")
    symbols = stocks.symbol.str.strip().tolist()
    if len(set(symbols)) != len(symbols) or any(not re.fullmatch(r"[0-9A-Z]+\.T", s) for s in symbols):
        raise ValueError("銘柄コードに重複、空文字または日本株コード以外があります。")
    return symbols + ["^N225"]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_saved(path, conditions):
    metadata = path.with_suffix(".json")
    if not metadata.exists():
        raise ValueError("既存CSVの取得条件記録がありません。上書きせず停止します。")
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    if saved["conditions"] != conditions or saved["sha256"] != file_hash(path):
        raise ValueError("既存CSVの取得条件またはファイル内容が記録と一致しません。")
    frame = pd.read_csv(path, index_col="Datetime")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert("Asia/Tokyo")
    return validate_data(frame, pd.Timestamp(conditions["start"]), pd.Timestamp(conditions["end"]), symbol=conditions["symbol"])


def fetch_data(symbol, start, end, retries, cooldown):
    for attempt in range(retries + 1):
        try:
            return yf.Ticker(symbol).history(
                start=start, end=end, interval="5m", auto_adjust=False,
                actions=False, keepna=True, timeout=30,
            )
        except YFRateLimitError:
            if attempt == retries:
                raise
            delay = cooldown * (attempt + 1)
            print(f"制限エラー: 全取得を{delay:g}秒停止、再試行{attempt + 1}/{retries}", flush=True)
            time.sleep(delay)


def run_setup(symbols, start, end, output, interval, retries, cooldown):
    output.mkdir(parents=True, exist_ok=True)
    report = output / f"setup_{pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y%m%d_%H%M%S_%f')}.json"
    results = []
    requested = False
    hide_errors = yf.config.debug.hide_exceptions
    yf.config.debug.hide_exceptions = False
    try:
        for i, symbol in enumerate(symbols):
            path = output / f"{symbol}_5m.csv"
            conditions = dict(symbol=symbol, start=start.isoformat(), end=end.isoformat(), interval="5m", auto_adjust=False, actions=False, keepna=True)
            result = dict(symbol=symbol, path=str(path), conditions=conditions)
            stop = False
            try:
                if path.exists():
                    frame = load_saved(path, conditions)
                    result["status"] = "reused"
                else:
                    if path.with_suffix(".json").exists():
                        raise ValueError("CSVのない取得条件記録が存在します。上書きせず停止します。")
                    if requested:
                        time.sleep(interval)
                    requested = True
                    frame = fetch_data(symbol, start, end, retries, cooldown)
                    frame = validate_data(frame, start, end, symbol=symbol)
                    save_csv(frame, path)
                    metadata = dict(conditions=conditions, retrieved_at=pd.Timestamp.now(tz="Asia/Tokyo").isoformat(), rows=len(frame), first=frame.index[0].isoformat(), last=frame.index[-1].isoformat(), sha256=file_hash(path), yfinance_version=yf.__version__, coverage="unverified")
                    write_json(metadata, path.with_suffix(".json"))
                    frame = load_saved(path, conditions)
                    result["status"] = "saved"
                result.update(rows=len(frame), first=frame.index[0].isoformat(), last=frame.index[-1].isoformat(), coverage="unverified")
            except YFRateLimitError as error:
                result.update(status="failed", reason=str(error))
                stop = True
            except Exception as error:
                result.update(status="failed", reason=f"{type(error).__name__}: {error}")
            results.append(result)
            print(f"[{i + 1}/{len(symbols)}] {symbol}: {result['status']} {result.get('rows', '')} {result.get('reason', '')}", flush=True)
            if stop:
                print("制限エラーが続くため、残りの取得を停止します。", flush=True)
                break
    finally:
        yf.config.debug.hide_exceptions = hide_errors
        remaining = symbols[len(results):]
        results.extend(dict(symbol=symbol, status="not_attempted") for symbol in remaining)
        write_json(dict(created_at=pd.Timestamp.now(tz="Asia/Tokyo").isoformat(), settings=dict(interval=interval, retries=retries, cooldown=cooldown), results=results), report)
    counts = {status: sum(r["status"] == status for r in results) for status in ("saved", "reused", "failed", "not_attempted")}
    print(f"保存={counts['saved']} 再利用={counts['reused']} 失敗={counts['failed']} 未取得={counts['not_attempted']}")
    print(f"結果記録: {report}")
    print("保存・再利用できたCSVの値・日時は検証済み。休場・無約定・配信欠損の判別と期間全体の網羅性は未確認です。")
    return 1 if counts["failed"] or counts["not_attempted"] else 0


def main():
    parser = argparse.ArgumentParser(description="tickers.csv全銘柄と日経平均の5分足を逐次取得します。")
    parser.add_argument("--start", default="earliest", help="開始日、またはearliest（直近60日以内の最古の丸一日）")
    parser.add_argument("--end", default="2026-10-02", help="終了日、この日を含む")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--cooldown", type=float, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--download", action="store_true", help="実際に取得する。指定なしでは条件確認だけ")
    args = parser.parse_args()
    try:
        if args.retries < 0 or any(not math.isfinite(v) or v <= 0 for v in (args.interval, args.cooldown)):
            raise ValueError("待機秒数は正の有限値、再試行回数は0以上にしてください。")
        symbols = read_symbols(ROOT / "tickers.csv")
        start, end = resolve_period(args.start, args.end, pd.Timestamp.now(tz="Asia/Tokyo"))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    output = (ROOT / args.output).resolve()
    print(f"対象={len(symbols)}件（個別株{len(symbols) - 1}、日経平均1） {start} 以上〜{end} 未満、5分足")
    print(f"保存先={output} 銘柄間={args.interval:g}秒 制限時={args.cooldown:g}秒×再試行回数 再試行上限={args.retries}")
    print(f"通常待機だけで約{(len(symbols) - 1) * args.interval / 60:.1f}分＋通信時間。制限時は延長します。")
    if not args.download:
        print("条件確認のみ。取得するには同じ指定に--downloadを付けてください。")
        return 0
    return run_setup(symbols, start, end, output, args.interval, args.retries, args.cooldown)


if __name__ == "__main__":
    raise SystemExit(main())
