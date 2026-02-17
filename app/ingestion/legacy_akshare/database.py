import sqlite3
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_path="data/database.db"):
        Path("data").mkdir(exist_ok=True)

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def create_tables(self):
        """
        创建生产级K线表
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS kline_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            trade_time TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            UNIQUE(ts_code, trade_time, timeframe)
        );
        """

        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_kline_lookup
        ON kline_data (ts_code, trade_time, timeframe);
        """

        self.cursor.execute(create_table_sql)
        self.cursor.execute(create_index_sql)
        self.conn.commit()

    def insert_kline_batch(self, data_list):
        """
        批量插入数据
        data_list: [(ts_code, trade_time, timeframe, open, high, low, close, volume), ...]
        """
        insert_sql = """
        INSERT OR IGNORE INTO kline_data
        (ts_code, trade_time, timeframe, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.cursor.executemany(insert_sql, data_list)
        self.conn.commit()

    def fetch_by_stock(self, ts_code, timeframe):
        query = """
        SELECT * FROM kline_data
        WHERE ts_code = ? AND timeframe = ?
        ORDER BY trade_time ASC;
        """
        self.cursor.execute(query, (ts_code, timeframe))
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()
