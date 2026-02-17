from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

KLT_MAP = {
    "1min": "1",
    "5min": "5",
    "60min": "60",
    "daily": "101",
}


@dataclass
class CrawlConfig:
    db_path: Path
    start_date: str
    end_date: str
    chunk_days: int = 365
    incremental: bool = True
    max_retries: int = 5
    connect_timeout: float = 5.0
    read_timeout: float = 20.0
    backoff_base: float = 0.6
    sleep_min: float = 0.3
    sleep_max: float = 0.8
    progress_every: int = 50
    failed_log_path: Path = Path("failed_symbols.log")
    run_log_path: Path = Path("crawler.log")


def today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")
