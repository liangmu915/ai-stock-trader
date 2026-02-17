# ml_stock_project

本目录是主业务项目，包含：

- 日频选股模型训练（LightGBM）
- 事件驱动回测
- PIT（Point-in-Time）预测
- Streamlit UI 控制台

---

## 主要入口

- UI：`ui_app.py`
- 命令行训练回测：`main.py`
- 命令行 PIT 预测：`pit_main.py`

---

## 目录说明

```text
ml_stock_project/
├─ config.py
├─ main.py
├─ pit_main.py
├─ ui_app.py
├─ jobs/
│  ├─ train_job.py          # UI 后台训练任务（可停止）
│  └─ feature_job.py        # UI 后台特征工程任务（可停止）
├─ src/
│  ├─ data_io.py
│  ├─ cleaning.py
│  ├─ features.py
│  ├─ split.py
│  ├─ train.py
│  ├─ backtest.py
│  ├─ evaluation.py
│  ├─ pit.py
│  ├─ metrics.py
│  └─ plots.py
├─ docs/
│  ├─ ARCHITECTURE.md
│  └─ UI_GUIDE.md
└─ outputs/
```

---

## 运行方式

### 1) 启动 UI

```bash
streamlit run ui_app.py
```

### 2) 命令行训练

```bash
python main.py
```

### 3) 命令行 PIT 预测

```bash
python pit_main.py
```

---

## 数据要求

默认读取 SQLite（建议 `../data/market_kline.db`），支持表：

- `kline_data`（推荐，抓取器持续更新）
- `daily_prices`（历史兼容）
- `stock_info`（股票名称/行业/市场等基础信息）

---

## 产物输出

- 模型文件：`../models/lgbm_daily_model_*.txt`
- 特征清单：`../models/feature_columns_*.csv`
- 最新软链接风格文件：
  - `../models/lgbm_daily_model_latest.txt`
  - `../models/feature_columns_latest.csv`
- 图表与评估：`outputs/`

---

## 详细文档

- 架构：`docs/ARCHITECTURE.md`
- UI 手册：`docs/UI_GUIDE.md`

