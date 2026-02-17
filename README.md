# AI A股量化系统（总览）

本仓库包含两条主线：

1. `ml_stock_project/`  
模型训练、回测、PIT 预测、Streamlit 可视化控制台。

2. `app/ingestion/`  
东方财富数据抓取模块（支持增量、限流重试、写入 SQLite）。

---

## 目录结构

```text
.
├─ app/
│  └─ ingestion/                 # 抓取子项目（生产抓取入口在 run_ingestion.py）
├─ data/                         # 数据库目录（推荐使用）
├─ ml_stock_project/             # 训练/回测/UI 主项目
├─ models/                       # 训练出的模型与特征清单
├─ notebooks/                    # 可选分析笔记
├─ requirements.txt              # 根环境依赖
└─ README.md
```

---

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 UI

```bash
cd ml_stock_project
streamlit run ui_app.py
```

当前 UI Tab 顺序：

1. 数据抓取
2. 数据可视化
3. 特征工程
4. 模型训练
5. 模型预测

---

## 文档导航

- 架构说明：`ml_stock_project/docs/ARCHITECTURE.md`
- UI 与模块使用说明：`ml_stock_project/docs/UI_GUIDE.md`
- ML 子项目说明：`ml_stock_project/README.md`
- 抓取子项目说明：`app/README.md`

---

## 推荐数据路径

- 日线数据库：`data/market_kline.db`
- 模型输出目录：`models/`
- 训练/回测输出目录：`ml_stock_project/outputs/`

---

## 发布到 GitHub 前建议

1. 清理临时输出  
可选删除 `ml_stock_project/outputs/*.log`、大体积中间文件。

2. 检查敏感信息  
确认 `.env`、token、私钥不在提交中。

3. 记录版本信息  
建议在 Release 说明中写明：
数据日期范围、训练参数、模型文件名、回测区间。

