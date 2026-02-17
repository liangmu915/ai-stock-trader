import logging
import random
import time
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple

import pandas as pd
import requests

try:
    from crawler.config import CrawlConfig, KLT_MAP
    from crawler.parser import parse_kline_rows
except ModuleNotFoundError:
    from config import CrawlConfig, KLT_MAP
    from parser import parse_kline_rows

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
MARKET_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"


def ts_code_to_secid(ts_code: str) -> str:
    code, market = ts_code.upper().split(".", 1)
    if market == "SZ":
        return f"0.{code}"
    if market == "SH":
        return f"1.{code}"
    raise ValueError(f"Unsupported market in ts_code: {ts_code}")


def _market_flag_to_ts_suffix(flag: str) -> str:
    if flag == "0":
        return "SZ"
    if flag == "1":
        return "SH"
    return ""


def _normalize_list_date(raw_value: str) -> str:
    text = str(raw_value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return ""


def dt_to_yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def split_date_range(start_date: str, end_date: str, chunk_days: int) -> List[Tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    chunks: List[Tuple[str, str]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        chunks.append((dt_to_yyyymmdd(cur), dt_to_yyyymmdd(chunk_end)))
        cur = chunk_end + timedelta(days=1)
    return chunks


class EastmoneyFetcher:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.logger = logging.getLogger("crawler.fetcher")

    def close(self) -> None:
        self.session.close()

    def _sleep_between_requests(self) -> None:
        delay = random.uniform(self.config.sleep_min, self.config.sleep_max)
        time.sleep(delay)

    def _request_with_retry(self, url: str, params: dict) -> dict:
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    timeout=(self.config.connect_timeout, self.config.read_timeout),
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout as exc:
                self.logger.warning("Timeout (attempt %s/%s): %s", attempt, self.config.max_retries, exc)
            except requests.RequestException as exc:
                self.logger.warning("Request error (attempt %s/%s): %s", attempt, self.config.max_retries, exc)

            if attempt < self.config.max_retries:
                backoff = self.config.backoff_base * (2 ** (attempt - 1))
                time.sleep(backoff + random.uniform(0, 0.3))

        raise RuntimeError("Request failed after maximum retries")

    def fetch_kline_chunk(self, ts_code: str, frequency: str, beg: str, end: str) -> pd.DataFrame:
        secid = ts_code_to_secid(ts_code)
        params = {
            "secid": secid,
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": KLT_MAP[frequency],
            "fqt": "1",
            "beg": beg,
            "end": end,
            "lmt": "1000000",
        }
        data = self._request_with_retry(KLINE_URL, params)
        payload = data.get("data") if isinstance(data, dict) else None
        klines = payload.get("klines") if isinstance(payload, dict) else None
        if not klines:
            return pd.DataFrame()
        return parse_kline_rows(ts_code=ts_code, frequency=frequency, klines=klines)

    def fetch_kline_segmented(self, ts_code: str, frequency: str, start_date: str, end_date: str) -> pd.DataFrame:
        chunks = split_date_range(start_date=start_date, end_date=end_date, chunk_days=self.config.chunk_days)
        all_parts: List[pd.DataFrame] = []
        for beg, end in chunks:
            try:
                part = self.fetch_kline_chunk(ts_code=ts_code, frequency=frequency, beg=beg, end=end)
                if not part.empty:
                    all_parts.append(part)
            finally:
                self._sleep_between_requests()

        if not all_parts:
            return pd.DataFrame()

        df = pd.concat(all_parts, ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code", "datetime", "frequency"]).sort_values("datetime")
        return df

    def fetch_all_market_ts_codes(self) -> List[str]:
        symbols: List[str] = []
        page = 1
        while True:
            params = {
                "pn": str(page),
                "pz": "1000",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f12",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
                "fields": "f12,f13",
            }
            data = self._request_with_retry(MARKET_URL, params)
            payload = data.get("data") if isinstance(data, dict) else None
            diff = payload.get("diff") if isinstance(payload, dict) else None
            if not diff:
                break

            for item in diff:
                code = str(item.get("f12", "")).strip()
                market_flag = str(item.get("f13", "")).strip()
                if len(code) != 6 or not code.isdigit():
                    continue
                if market_flag == "0":
                    symbols.append(f"{code}.SZ")
                elif market_flag == "1":
                    symbols.append(f"{code}.SH")

            self._sleep_between_requests()
            page += 1

        uniq = sorted(set(symbols))
        self.logger.info("Fetched %d A-share symbols from Eastmoney list API", len(uniq))
        return uniq

    def fetch_all_market_stock_info(self) -> pd.DataFrame:
        """
        Fetch stock basic info from Eastmoney list API.
        Returns columns:
        ts_code, symbol, name, market, industry, list_date
        """
        rows_out = []
        page = 1
        while True:
            params = {
                "pn": str(page),
                "pz": "1000",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f12",
                "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
                # f12 code, f13 market flag, f14 name, f100 industry, f26 list date (best-effort)
                "fields": "f12,f13,f14,f100,f26",
            }
            data = self._request_with_retry(MARKET_URL, params)
            payload = data.get("data") if isinstance(data, dict) else None
            diff = payload.get("diff") if isinstance(payload, dict) else None
            if not diff:
                break

            for item in diff:
                code = str(item.get("f12", "")).strip()
                market_flag = str(item.get("f13", "")).strip()
                suffix = _market_flag_to_ts_suffix(market_flag)
                if len(code) != 6 or not code.isdigit() or not suffix:
                    continue
                ts_code = f"{code}.{suffix}"
                name = str(item.get("f14", "") or "").strip()
                industry = str(item.get("f100", "") or "").strip()
                list_date = _normalize_list_date(str(item.get("f26", "") or ""))
                rows_out.append(
                    {
                        "ts_code": ts_code,
                        "symbol": code,
                        "name": name,
                        "market": suffix,
                        "industry": industry,
                        "list_date": list_date,
                    }
                )

            self._sleep_between_requests()
            page += 1

        if not rows_out:
            return pd.DataFrame()
        df = pd.DataFrame(rows_out).drop_duplicates(subset=["ts_code"]).sort_values("ts_code")
        self.logger.info("Fetched %d stock_info rows from Eastmoney", len(df))
        return df


def load_symbols_from_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip().upper() for x in line.split(",") if x.strip()]
            out.extend(parts)
    return sorted(set(out))


def normalize_symbols(symbols: Iterable[str]) -> List[str]:
    out = [s.strip().upper() for s in symbols if s.strip()]
    return sorted(set(out))
