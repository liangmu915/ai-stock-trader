import argparse
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from crawler.config import CrawlConfig, KLT_MAP, today_yyyymmdd
    from crawler.database import (
        get_max_datetime,
        init_db,
        insert_batch,
        refresh_features_for_symbol_frequency,
        upsert_stock_info_batch,
    )
    from crawler.fetcher import EastmoneyFetcher, load_symbols_from_file, normalize_symbols
except ModuleNotFoundError:
    from config import CrawlConfig, KLT_MAP, today_yyyymmdd
    from database import (
        get_max_datetime,
        init_db,
        insert_batch,
        refresh_features_for_symbol_frequency,
        upsert_stock_info_batch,
    )
    from fetcher import EastmoneyFetcher, load_symbols_from_file, normalize_symbols


def setup_logging(run_log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_log_path, encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production Eastmoney historical kline crawler")

    parser.add_argument("--single", type=str, help="Single symbol, e.g. 000001.SZ")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--symbols-file", type=str, help="File with symbols")
    parser.add_argument("--full-market", action="store_true", help="Fetch full A-share list from Eastmoney")

    parser.add_argument("--frequencies", type=str, default="1min,5min,60min,daily")
    parser.add_argument("--start-date", type=str, default="19980101")
    parser.add_argument("--end-date", type=str, default=today_yyyymmdd())
    parser.add_argument("--incremental", action="store_true", help="Incremental mode")

    parser.add_argument("--db-path", type=str, default="market_kline.db")
    parser.add_argument("--chunk-days", type=int, default=365)

    parser.add_argument("--sleep-min", type=float, default=0.3)
    parser.add_argument("--sleep-max", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=20.0)
    parser.add_argument("--backoff-base", type=float, default=0.6)

    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--failed-log", type=str, default="failed_symbols.log")
    parser.add_argument("--run-log", type=str, default="crawler.log")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature_data calculation/update")
    parser.add_argument("--feature-lookback", type=int, default=300, help="Lookback rows for feature recompute")
    parser.add_argument("--refresh-stock-info", action="store_true", help="Fetch and upsert stock_info from Eastmoney")

    return parser.parse_args()


def build_symbol_list(args: argparse.Namespace, fetcher: EastmoneyFetcher) -> List[str]:
    symbols: List[str] = []
    if args.single:
        symbols.append(args.single)
    if args.symbols:
        symbols.extend([x.strip() for x in args.symbols.split(",") if x.strip()])
    if args.symbols_file:
        symbols.extend(load_symbols_from_file(args.symbols_file))
    if args.full_market:
        symbols.extend(fetcher.fetch_all_market_ts_codes())

    symbols = normalize_symbols(symbols)
    if not symbols:
        raise ValueError("No symbols provided. Use --single, --symbols, --symbols-file, or --full-market")
    return symbols


def parse_frequencies(text: str) -> List[str]:
    freqs = [x.strip() for x in text.split(",") if x.strip()]
    invalid = [f for f in freqs if f not in KLT_MAP]
    if invalid:
        raise ValueError(f"Invalid frequencies: {invalid}. Valid: {list(KLT_MAP.keys())}")
    return freqs


def yyyymmdd_from_db_dt(db_dt: str) -> str:
    return datetime.strptime(db_dt, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d")


def append_failed(failed_log_path: Path, ts_code: str, frequency: str, reason: str) -> None:
    with open(failed_log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\t{ts_code}\t{frequency}\t{reason}\n")


def run() -> None:
    args = parse_args()
    config = CrawlConfig(
        db_path=Path(args.db_path),
        start_date=args.start_date,
        end_date=args.end_date,
        chunk_days=args.chunk_days,
        incremental=args.incremental,
        max_retries=args.max_retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        backoff_base=args.backoff_base,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        progress_every=args.progress_every,
        failed_log_path=Path(args.failed_log),
        run_log_path=Path(args.run_log),
    )

    setup_logging(config.run_log_path)
    logger = logging.getLogger("crawler.main")

    start_ts = time.time()
    total_inserted = 0
    total_failed = 0
    total_feature_upserts = 0
    stock_info_upserts = 0
    freq_stats: Dict[str, int] = {}

    fetcher = EastmoneyFetcher(config)

    with sqlite3.connect(config.db_path) as conn:
        init_db(conn)
        if args.refresh_stock_info or args.full_market:
            try:
                info_df = fetcher.fetch_all_market_stock_info()
                stock_info_upserts = upsert_stock_info_batch(conn, info_df)
                logger.info("Stock info upserted=%d", stock_info_upserts)
            except Exception as exc:
                logger.error("Failed to refresh stock_info: %s", exc)

        symbols = build_symbol_list(args, fetcher)
        freqs = parse_frequencies(args.frequencies)

        logger.info("Start crawling: symbols=%d, frequencies=%s", len(symbols), freqs)

        for idx, ts_code in enumerate(symbols, start=1):
            symbol_inserted = 0
            symbol_feature_upserts = 0
            per_freq_inserted: Dict[str, int] = {}
            for frequency in freqs:
                try:
                    req_start = config.start_date
                    max_dt = None
                    if config.incremental:
                        max_dt = get_max_datetime(conn, ts_code, frequency)
                        if max_dt:
                            req_start = max(req_start, yyyymmdd_from_db_dt(max_dt))

                    df = fetcher.fetch_kline_segmented(
                        ts_code=ts_code,
                        frequency=frequency,
                        start_date=req_start,
                        end_date=config.end_date,
                    )

                    if not df.empty and max_dt:
                        df = df[df["datetime"] > max_dt]

                    inserted = insert_batch(conn, df)
                    symbol_inserted += inserted
                    total_inserted += inserted
                    per_freq_inserted[frequency] = per_freq_inserted.get(frequency, 0) + inserted
                    freq_stats[frequency] = freq_stats.get(frequency, 0) + inserted

                    if not args.skip_features:
                        feature_upserts = refresh_features_for_symbol_frequency(
                            conn=conn,
                            ts_code=ts_code,
                            frequency=frequency,
                            lookback_rows=args.feature_lookback,
                        )
                        symbol_feature_upserts += feature_upserts
                        total_feature_upserts += feature_upserts
                except Exception as exc:
                    total_failed += 1
                    logger.error("Failed %s %s: %s", ts_code, frequency, exc)
                    append_failed(config.failed_log_path, ts_code, frequency, str(exc))
                    continue

            freq_text = ", ".join(f"{k}:{v}" for k, v in per_freq_inserted.items()) if per_freq_inserted else "-"
            logger.info(
                "Done %d/%d %s: inserted=%d, feature_upserted=%d, by_freq=[%s], db=%s",
                idx,
                len(symbols),
                ts_code,
                symbol_inserted,
                symbol_feature_upserts,
                freq_text,
                config.db_path,
            )

            if idx % config.progress_every == 0 or idx == len(symbols):
                logger.info(
                    "Progress: %d/%d symbols processed, inserted=%d, failed=%d",
                    idx,
                    len(symbols),
                    total_inserted,
                    total_failed,
                )

    fetcher.close()

    elapsed = time.time() - start_ts
    logger.info("Finished. elapsed=%.1fs, inserted=%d, failed=%d", elapsed, total_inserted, total_failed)
    logger.info("Feature rows upserted=%d", total_feature_upserts)
    logger.info("Stock info rows upserted=%d", stock_info_upserts)
    for freq, cnt in freq_stats.items():
        logger.info("Frequency summary: %s inserted=%d", freq, cnt)
    logger.info("DB=%s, failed_log=%s", config.db_path, config.failed_log_path)


if __name__ == "__main__":
    run()
