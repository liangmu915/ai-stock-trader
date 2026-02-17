# Crawler Usage

This crawler supports Eastmoney historical kline for:
- 1min
- 5min
- 60min
- daily

## Files
- `crawler/config.py`
- `crawler/fetcher.py`
- `crawler/parser.py`
- `crawler/database.py`
- `crawler/main.py`

## Examples

Single stock + two frequencies:

```bash
python crawler/main.py --single 000001.SZ --frequencies 5min,daily --start-date 20240101 --end-date 20260216 --incremental
# or:
python -m crawler.main --single 000001.SZ --frequencies 5min,daily --start-date 20240101 --end-date 20260216 --incremental
```

From symbol list file:

```bash
python crawler/main.py --symbols-file symbols.txt --frequencies 1min,5min,60min,daily --start-date 20250101 --end-date 20260216 --incremental
```

Full market crawl:

```bash
python crawler/main.py --full-market --frequencies daily --start-date 20100101 --end-date 20260216 --incremental --progress-every 100
```

## Key options
- `--incremental`: only fetch data newer than max datetime in DB for stock+frequency
- `--chunk-days`: segmented request window size (default 365)
- `--sleep-min/--sleep-max`: anti-rate-limit sleep range
- `--max-retries`: retry count with exponential backoff
- `--failed-log`: file to store failed symbol+frequency

## Database
- SQLite table: `kline_data`
- Unique index: `(ts_code, datetime, frequency)`
- Insert mode: `INSERT OR IGNORE` batch insert
- Extra raw fields: `amplitude`, `pct_change`, `change`, `turnover`

- SQLite table: `feature_data`
- Unique index: `(ts_code, datetime, frequency)`
- Upsert mode: `INSERT OR REPLACE`
- Feature columns:
  - `return_1`, `return_5`
  - `ma_5`, `ma_10`, `ma_20`
  - `ema_12`, `ema_26`, `dif`, `dea`, `macd`
  - `rsi_14`, `atr_14`, `vol_z_20`

This design is restart-safe. Rerun will continue from DB state.

## Feature control

- Default: crawler auto updates `feature_data`
- Skip features:

```bash
python crawler/main.py --single 000001.SZ --frequencies 5min --skip-features
```

## UI Dashboard

Visualize crawled data from SQLite:

```bash
streamlit run crawler/ui.py
```

In sidebar:
- set DB path (default `market_kline.db`)
- pick symbol/frequency/date range
- adjust max rows
