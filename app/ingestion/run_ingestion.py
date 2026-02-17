from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def _run_cmd(cmd: List[str], cwd: Path) -> int:
    print(f"[RUN] cwd={cwd}")
    print(f"[RUN] cmd={' '.join(shlex.quote(x) for x in cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    proc.wait()
    return int(proc.returncode or 0)


def _build_daily_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, "fetch_daily.py"]
    if args.db:
        cmd += ["--db", args.db]
    if args.csv:
        cmd += ["--csv", args.csv]
    if args.failed_log:
        cmd += ["--failed-log", args.failed_log]
    if args.years is not None:
        cmd += ["--years", str(args.years)]
    return cmd


def _build_crawler_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, "crawler/main.py"]
    if args.single:
        cmd += ["--single", args.single]
    if args.symbols:
        cmd += ["--symbols", args.symbols]
    if args.symbols_file:
        cmd += ["--symbols-file", args.symbols_file]
    if args.full_market:
        cmd += ["--full-market"]

    if args.frequencies:
        cmd += ["--frequencies", args.frequencies]
    if args.start_date:
        cmd += ["--start-date", args.start_date]
    if args.end_date:
        cmd += ["--end-date", args.end_date]
    if args.incremental:
        cmd += ["--incremental"]

    if args.db_path:
        cmd += ["--db-path", args.db_path]
    if args.chunk_days is not None:
        cmd += ["--chunk-days", str(args.chunk_days)]

    if args.sleep_min is not None:
        cmd += ["--sleep-min", str(args.sleep_min)]
    if args.sleep_max is not None:
        cmd += ["--sleep-max", str(args.sleep_max)]
    if args.max_retries is not None:
        cmd += ["--max-retries", str(args.max_retries)]
    if args.connect_timeout is not None:
        cmd += ["--connect-timeout", str(args.connect_timeout)]
    if args.read_timeout is not None:
        cmd += ["--read-timeout", str(args.read_timeout)]
    if args.backoff_base is not None:
        cmd += ["--backoff-base", str(args.backoff_base)]

    if args.progress_every is not None:
        cmd += ["--progress-every", str(args.progress_every)]
    if args.failed_log:
        cmd += ["--failed-log", args.failed_log]
    if args.run_log:
        cmd += ["--run-log", args.run_log]

    if args.skip_features:
        cmd += ["--skip-features"]
    if args.feature_lookback is not None:
        cmd += ["--feature-lookback", str(args.feature_lookback)]
    if args.refresh_stock_info:
        cmd += ["--refresh-stock-info"]

    return cmd


def _add_daily_parser(subparsers) -> None:
    p = subparsers.add_parser("daily", help="Run daily downloader (fetch_daily.py)")
    p.add_argument("--db", type=str, default="market_kline.db", help="SQLite DB path")
    p.add_argument("--csv", type=str, default=None, help="Optional symbol CSV (must contain ts_code)")
    p.add_argument("--failed-log", type=str, default="failed_symbols.txt", help="Failed symbol log")
    p.add_argument("--years", type=int, default=15, help="Fetch latest N years")


def _add_crawler_parser(subparsers) -> None:
    p = subparsers.add_parser("kline", help="Run multi-frequency crawler (crawler/main.py)")
    p.add_argument("--single", type=str, help="Single symbol, e.g. 000001.SZ")
    p.add_argument("--symbols", type=str, help="Comma-separated symbols")
    p.add_argument("--symbols-file", type=str, help="Symbol file")
    p.add_argument("--full-market", action="store_true", help="Fetch all symbols")

    p.add_argument("--frequencies", type=str, default="1min,5min,60min,daily")
    p.add_argument("--start-date", type=str, default="19980101")
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--incremental", action="store_true")

    p.add_argument("--db-path", type=str, default="market_kline.db")
    p.add_argument("--chunk-days", type=int, default=365)

    p.add_argument("--sleep-min", type=float, default=0.3)
    p.add_argument("--sleep-max", type=float, default=0.8)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--connect-timeout", type=float, default=5.0)
    p.add_argument("--read-timeout", type=float, default=20.0)
    p.add_argument("--backoff-base", type=float, default=0.6)

    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--failed-log", type=str, default="failed_symbols.log")
    p.add_argument("--run-log", type=str, default="crawler.log")
    p.add_argument("--skip-features", action="store_true")
    p.add_argument("--feature-lookback", type=int, default=300)
    p.add_argument("--refresh-stock-info", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ingestion entry point")
    parser.add_argument(
        "--workdir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Working directory for ingestion scripts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_daily_parser(subparsers)
    _add_crawler_parser(subparsers)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        print(f"[ERROR] Workdir not found: {workdir}")
        return 2

    if args.command == "daily":
        cmd = _build_daily_cmd(args)
    elif args.command == "kline":
        cmd = _build_crawler_cmd(args)
    else:
        print(f"[ERROR] Unknown command: {args.command}")
        return 2

    code = _run_cmd(cmd, cwd=workdir)
    if code == 0:
        print("[OK] Ingestion task completed")
    else:
        print(f"[FAIL] Ingestion task failed, exit_code={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
