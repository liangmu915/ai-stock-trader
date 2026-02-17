# UI 与模块使用说明

本文档给出 UI 的完整操作流程，以及与底层模块的关系。

---

## 1. 启动方式

```bash
cd ml_stock_project
streamlit run ui_app.py
```

---

## 2. Tab 说明（按顺序）

### 2.1 数据抓取

用途：从东方财富抓取数据并增量写入 SQLite。

关键参数：

- 目标数据库（建议 `../data/market_kline.db`）
- 抓取频率（`daily`/`1min`/`5min`/`60min`/组合）
- 开始日期、截止日期
- 股票范围（单只/多只/全市场）
- 超时、重试、sleep、是否跳过特征回写

输出：

- 日志显示每只股票完成信息
- 数据写入 `kline_data`
- 失败记录 `failed_symbols.log`

停止机制：

- 点击“停止抓取”，UI 会终止后台子进程。

---

### 2.2 数据可视化

用途：检查数据库覆盖范围与个股 K 线。

功能：

- 选择数据库并加载
- 代码/名称搜索股票
- 快捷区间（7日/30日/90日/1年/5年）
- 自定义日期区间
- K 线 + 成交量图
- 显示股票名称、行业、市场、可用行数、覆盖年数

注意：

- 可视化优先读 `kline_data`，再回退 `daily_prices`。
- 若看不到新数据，先点“加载数据库”刷新缓存。

---

### 2.3 特征工程

用途：从 raw 数据库生成 engineered 特征库。

流程：

1. 选择模型组合（读取其 `feature_columns_*.csv`）
2. 选择 raw 数据库
3. 指定输出 engineered 数据库路径
4. 执行特征工程

结果：

- 输出表：`engineered_features`
- 元信息表：`engineered_meta`

进度：

- 显示阶段进度、已耗时、ETA、关键行数日志。

停止：

- 点击“停止特征工程”。

---

### 2.4 模型训练

用途：训练 LightGBM 并输出回测与评估结果。

支持两种训练来源：

1. `raw数据库（现算特征）`
2. `engineered数据库（直接训练）`

参数：

- 时间切分（train/valid/test）
- LightGBM 参数
- TopK、行业持仓上限 `max_per_sector`

输出：

- `models/lgbm_daily_model_*.txt`
- `models/feature_columns_*.csv`
- `outputs/` 下图表和 CSV

进度：

- 阶段进度、已耗时、ETA、关键行数日志。

停止：

- 点击“停止训练”。

---

### 2.5 模型预测

用途：读取现有模型，对目标日期做推荐。

流程：

1. 选择模型和特征清单
2. 选择数据库类型（raw / engineered）
3. 选择数据库
4. 输入目标日期（可留空用最近交易日）
5. 设置 TopK 和 `max_per_sector`

输出：

- TopK 推荐表
- CSV 下载

进度：

- 显示预测阶段进度（加载数据、准备日期、执行预测、输出结果）与耗时。

---

## 3. 常见问题

### Q1：抓取了很多数据，但可视化行数没变

原因通常是：

- 还在看旧库路径；或
- 页面缓存未刷新；或
- 区间/最大读取行数限制。

处理：

1. 在“数据可视化”点击“加载数据库”
2. 用“自定义”区间拉到更早日期
3. 把“最大读取行数”调大

### Q2：为什么全市场抓取速度慢

原因包括：

- 网络重试与退避
- 全市场标的数量大
- 特征回写开启时 I/O 较重

建议：

- 全市场抓取时勾选“跳过特征回写”
- 使用增量模式
- 适当提高 read timeout

### Q3：engineered 数据还会在训练时再清洗吗

不会。  
清洗主要发生在特征工程阶段；训练使用 engineered 时直接进入训练流程。

---

## 4. 与底层模块对应关系

- 抓取页 -> `app/ingestion/run_ingestion.py` -> `app/ingestion/crawler/*`
- 特征工程页 -> `jobs/feature_job.py` -> `src/data_io.py` / `src/cleaning.py` / `src/features.py`
- 训练页 -> `jobs/train_job.py` -> `src/train.py` / `src/backtest.py` / `src/evaluation.py`
- 预测页 -> `src/pit.py`
- 可视化页 -> `ui_app.py` 内 SQLite 查询与 Plotly 渲染

