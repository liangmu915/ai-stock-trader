# Eastmoney 5-Minute Fetcher

A small Python project to fetch free 5-minute A-share historical K-line data from Eastmoney and save it into SQLite.

## Features

- Uses Eastmoney API endpoint:
  - `https://push2his.eastmoney.com/api/qt/stock/kline/get`
- Supports TS codes like `000001.SZ`, `600000.SH`
- Converts TS code to Eastmoney `secid`
- Parses data into columns:
  - `ts_code, datetime, open, high, low, close, volume, amount`
- Saves into local SQLite database `market.db` table `minute_5m`
- Uses unique index `(ts_code, datetime)` to avoid duplicates
- Incremental update:
  - reads max datetime from DB and fetches only newer bars

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run

```bash
python main.py --symbols 000001.SZ,600000.SH --beg 20250101 --end 20250131 --db market.db --sleep 0.5
```

3. Check how much data is available for a period (e.g. 60min)

```bash
python coverage_check.py --symbols 000001.SZ,600000.SH --klt 60
```

4. Check AkShare coverage with the same output format

```bash
python ak_coverage_check.py --symbols 000001.SZ,600000.SH --klt 60
python ak_coverage_check.py --symbols 000001.SZ,600000.SH --klt 101
python ak_coverage_check.py --symbols 000001.SZ,600000.SH --klt 1
```

Default behavior is best-effort max data:
- If the requested window has no 5m bars, the script falls back to all available 5m bars returned by API.
- Use `--strict-window` if you want to disable fallback.

## Parameters

- `--symbols`: comma-separated TS codes
- `--beg`: begin date `YYYYMMDD`
- `--end`: end date `YYYYMMDD`
- `--db`: sqlite file path
- `--sleep`: sleep seconds between requests
- `--strict-window`: disable best-effort fallback

## Notes

- `secid` mapping:
  - `SZ -> 0.{code}`
  - `SH -> 1.{code}`
- For 5-minute K-line, this script requests a wide server-side range (`beg=0`, `end=20500000`)
  and then filters by your `--beg` / `--end` locally. This is more reliable for Eastmoney minute API.
- API may return empty data for unsupported ranges or suspended symbols.
- The script uses soft error handling and continues with next symbol.
