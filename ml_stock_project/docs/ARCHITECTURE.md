# 系统架构说明

本文档说明当前项目的核心架构、数据流、模块职责与关键设计。

---

## 1. 总体架构

系统分为两层：

1. 数据层（Ingestion）
2. 策略层（ML + Backtest + UI）

数据层负责从东方财富接口抓取并入库；策略层负责特征工程、训练、回测、预测与可视化。

---

## 2. 数据层（app/ingestion）

### 2.1 数据来源

- 东方财富 K 线接口：
  - `https://push2his.eastmoney.com/api/qt/stock/kline/get`
- 市场股票列表接口：
  - `https://82.push2.eastmoney.com/api/qt/clist/get`

### 2.2 核心模块

- `crawler/fetcher.py`
  - API 请求、重试、超时、限流 sleep、分段抓取。
- `crawler/parser.py`
  - 原始 K 线字符串解析为结构化 DataFrame。
- `crawler/database.py`
  - SQLite 建表、去重写入、特征表写入。
- `crawler/main.py`
  - CLI 主流程、增量逻辑、进度与失败日志。
- `run_ingestion.py`
  - 统一入口（daily / kline）。

### 2.3 表结构（核心）

- `kline_data`
  - `ts_code, datetime, open, high, low, close, volume, amount, frequency, ...`
- `stock_info`
  - `ts_code, symbol, name, market, industry, list_date, updated_at`
- `feature_data`（抓取侧可选）

唯一索引：

- `kline_data(ts_code, datetime, frequency)`
- `feature_data(ts_code, datetime, frequency)`

---

## 3. 策略层（ml_stock_project）

### 3.1 处理流水线

1. `src/data_io.py`：从 SQLite/CSV/Parquet 读取并标准化字段。
2. `src/cleaning.py`：清洗数据（去重、非法值、首 60 日过滤等）。
3. `src/features.py`：构建技术特征 + 横截面 rank + 标签。
4. `src/split.py`：按日期切分 train/valid/test。
5. `src/train.py`：LightGBM 训练（AUC/early stopping）。
6. `src/backtest.py`：事件驱动回测（含涨停限制、持仓规则）。
7. `src/evaluation.py`：分组收益、年度收益、命中率、集中度。
8. `src/pit.py`：点时预测（读取训练模型进行当日推荐）。

### 3.2 无泄露原则

- 特征仅用历史窗口（rolling/shift 严格向后看）。
- 标签使用未来收益（`shift(-5)`），不参与特征构造。
- 回测按信号日与交易日错位，避免同日未来信息。

### 3.3 行业相关机制

- 可使用静态行业快照（`stock_info` + `INDUSTRY_SNAPSHOT_AT`）。
- 支持行业约束选股（`max_per_sector`）。
- 注意：静态快照不是完整 as-of 历史行业映射。

---

## 4. UI 架构（ui_app.py）

UI 当前有 5 个 Tab（按显示顺序）：

1. 数据抓取
2. 数据可视化
3. 特征工程
4. 模型训练
5. 模型预测

### 4.1 后台任务机制

训练、特征工程、抓取使用后台子进程，支持：

- 实时日志
- 进度条
- 已耗时与 ETA
- 停止任务（terminate/kill）

任务脚本：

- `jobs/train_job.py`
- `jobs/feature_job.py`

---

## 5. 数据读取优先级（可视化）

为避免旧表覆盖新数据，可视化优先读取：

1. `kline_data`（新抓取）
2. `daily_prices`（历史回退）

---

## 6. 模型与输出管理

模型输出到根目录 `models/`：

- `lgbm_daily_model_YYYYMMDD_HHMMSS.txt`
- `feature_columns_YYYYMMDD_HHMMSS.csv`
- `lgbm_daily_model_latest.txt`
- `feature_columns_latest.csv`

图表和评估输出到：

- `ml_stock_project/outputs/`

