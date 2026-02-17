from pathlib import Path

# 数据源配置
DATA_SOURCE = "sqlite"  # one of: csv, parquet, sqlite
DATA_PATH = Path("../market_kline.db")
SQLITE_TABLE = "daily_prices"  # fallback to kline_data(freq='daily') if missing

# 标准输入列
DATE_COL = "date"
STOCK_COL = "stock"

# 回测参数
TOPK = 10
HOLD_DAYS = 5
MAX_PER_SECTOR = 2

# 时间切分
TRAIN_START = "2009-01-01"
TRAIN_END = "2019-12-31"
VALID_START = "2020-01-01"
VALID_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"

# 其他配置
RANDOM_SEED = 42
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("../models")

# 行业映射模式：
# None -> 使用 stock_info 最新快照
# 例如 "2026-02-17 00:00:00" -> 仅使用 updated_at <= 该时间点的数据
INDUSTRY_SNAPSHOT_AT = None

# 快速测试样本量；设为 None 表示全市场
SAMPLE_STOCKS = None
