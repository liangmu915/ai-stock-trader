# app（数据抓取子项目）

本目录用于统一存放抓取相关模块，主入口在：

- `app/ingestion/run_ingestion.py`

---

## 目录说明

```text
app/
└─ ingestion/
   ├─ crawler/                    # 生产抓取器（Eastmoney）
   │  ├─ main.py
   │  ├─ fetcher.py
   │  ├─ parser.py
   │  ├─ database.py
   │  └─ config.py
   ├─ run_ingestion.py            # 统一抓取入口
   ├─ fetch_daily.py              # 历史日线脚本
   ├─ eastmoney_5m_fetcher/       # 早期实验
   └─ legacy_akshare/             # 兼容保留
```

---

## 统一抓取命令

### 1) 日线抓取脚本

```bash
python app/ingestion/run_ingestion.py daily --db data/market_kline.db --years 15
```

### 2) 多频率抓取（单只）

```bash
python app/ingestion/run_ingestion.py kline --single 000001.SZ --frequencies daily --start-date 19980101 --incremental --db-path data/market_kline.db
```

### 3) 全市场增量抓取（日线）

```bash
python app/ingestion/run_ingestion.py kline --full-market --frequencies daily --start-date 19980101 --incremental --db-path data/market_kline.db --skip-features
```

---

## 特性

- `requests.Session()` + 重试 + 指数退避
- 显式 connect/read timeout
- 分段抓取（chunk）
- 增量更新（按 `ts_code + frequency + datetime`）
- SQLite 去重写入
- 抓取失败记录到 `failed_symbols.log`

