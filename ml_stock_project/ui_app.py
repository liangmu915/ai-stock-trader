from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from src.backtest import backtest_topk
from src.cleaning import clean_data
from src.data_io import load_data, load_stock_info_snapshot
from src.evaluation import evaluate_predictions
from src.features import add_labels, build_features
from src.metrics import calc_auc
from src.pit import predict_topk_for_date_with_saved_model
from src.plots import plot_cumulative_return, plot_feature_importance, plot_score_distribution
from src.train import train_model


st.set_page_config(page_title="A-Share ML Control Panel", layout="wide")

BASE_COLS = ["date", "stock", "open", "high", "low", "close", "volume"]
INFO_COLS = ["name", "symbol", "market", "industry", "list_date"]
ENG_TABLE = "engineered_features"


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return bool(row)


def _kline_schema(conn: sqlite3.Connection) -> str:
    if not _table_exists(conn, "kline_data"):
        return "none"
    cols = [c[1] for c in conn.execute("PRAGMA table_info(kline_data)").fetchall()]
    if ("datetime" in cols) and ("frequency" in cols):
        return "new"
    if ("trade_time" in cols) and ("timeframe" in cols):
        return "legacy"
    return "unknown"


def _list_data_db_files() -> list[Path]:
    files: list[Path] = []
    for p in [Path("../data"), Path(".")]:
        if p.exists():
            files.extend(sorted(p.glob("*.db")))
    cfg_db = Path(config.DATA_PATH)
    if cfg_db.exists() and cfg_db not in files:
        files.append(cfg_db)
    # dedup
    out = []
    seen = set()
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            out.append(f)
            seen.add(k)
    return out


def _default_db_choice(db_opts: list[str]) -> tuple[int, str]:
    """
    统一默认数据库：优先 ../data/market_kline.db（即 /data/market_kline.db）。
    """
    preferred = str((Path("../data") / "market_kline.db").resolve())
    resolved_opts = [str(Path(x).resolve()) for x in db_opts]
    if preferred in resolved_opts:
        idx = resolved_opts.index(preferred)
        return idx, db_opts[idx]
    if db_opts:
        return 0, db_opts[0]
    return 0, ""


def _list_model_bundles() -> list[dict]:
    model_dir = Path(config.MODEL_DIR)
    if not model_dir.exists():
        return []
    models = sorted(model_dir.glob("lgbm_daily_model_*.txt"))
    bundles = []
    for m in models:
        if m.name == "lgbm_daily_model_latest.txt":
            feat = model_dir / "feature_columns_latest.csv"
        else:
            suffix = m.name.replace("lgbm_daily_model_", "").replace(".txt", "")
            feat = model_dir / f"feature_columns_{suffix}.csv"
        if feat.exists():
            bundles.append(
                {
                    "label": f"{m.name} | {feat.name}",
                    "model_path": str(m),
                    "feature_path": str(feat),
                }
            )
    # put latest on top if exists
    latest_m = model_dir / "lgbm_daily_model_latest.txt"
    latest_f = model_dir / "feature_columns_latest.csv"
    if latest_m.exists() and latest_f.exists():
        bundles.insert(
            0,
            {
                "label": f"{latest_m.name} | {latest_f.name} (推荐)",
                "model_path": str(latest_m),
                "feature_path": str(latest_f),
            },
        )
    return bundles


def _read_feature_list(feature_csv_path: str) -> list[str]:
    f = Path(feature_csv_path)
    if not f.exists():
        raise ValueError(f"特征文件不存在: {f}")
    df = pd.read_csv(f)
    if "feature" not in df.columns:
        raise ValueError("特征文件缺少列: feature")
    return df["feature"].dropna().astype(str).tolist()


def _prepare_feature_df_from_raw(db_path: str, sample_stocks: int | None) -> tuple[pd.DataFrame, list[str]]:
    df = load_data(
        data_source="sqlite",
        data_path=Path(db_path),
        sqlite_table=config.SQLITE_TABLE,
        sample_stocks=sample_stocks,
    )
    info = load_stock_info_snapshot(
        data_source="sqlite",
        data_path=Path(db_path),
        snapshot_at=config.INDUSTRY_SNAPSHOT_AT,
    )
    if not info.empty:
        df = df.merge(info, on="stock", how="left")
    df = clean_data(df)
    feat_df, feat_cols = build_features(df)
    return feat_df.sort_values(["date", "stock"]).reset_index(drop=True), feat_cols


def _load_engineered_df(db_path: str, table_name: str = ENG_TABLE) -> pd.DataFrame:
    with _connect(db_path) as conn:
        if not _table_exists(conn, table_name):
            raise ValueError(f"{table_name} 表不存在: {db_path}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    if "date" not in df.columns:
        raise ValueError(f"{table_name} 缺少 date 列")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "stock"]).sort_values(["date", "stock"]).reset_index(drop=True)
    return df


def _infer_feature_cols_from_engineered(df: pd.DataFrame) -> list[str]:
    exclude = set(BASE_COLS + INFO_COLS + ["prev_close", "ret_1", "future_return_5d", "label"])
    return [c for c in df.columns if c not in exclude]


def _validate_feature_requirements(df: pd.DataFrame, required_cols: list[str]) -> tuple[bool, list[str]]:
    miss = [c for c in required_cols if c not in df.columns]
    return len(miss) == 0, miss


def _split_by_custom_date(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
):
    d = pd.to_datetime(df["date"])
    train = df[(d >= train_start) & (d <= train_end)].copy()
    valid = df[(d >= valid_start) & (d <= valid_end)].copy()
    test = df[(d >= test_start) & (d <= test_end)].copy()
    return train, valid, test


def _log(msg: str, logs: list[str], holder) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    logs.append(f"[{ts}] {msg}")
    holder.code("\n".join(logs[-400:]), language="text")


def _task_is_running() -> bool:
    task = st.session_state.get("bg_task")
    if not task:
        return False
    proc = task.get("proc")
    return bool(proc is not None and proc.poll() is None)


def _start_bg_task(name: str, cmd: list[str], cwd: Path) -> None:
    if _task_is_running():
        raise RuntimeError("已有任务正在运行，请先停止当前任务。")
    log_dir = Path(config.OUTPUT_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{ts}.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    st.session_state["bg_task"] = {
        "name": name,
        "proc": proc,
        "cmd": cmd,
        "log_path": str(log_path),
        "start_ts": time.time(),
    }


def _stop_bg_task() -> tuple[bool, str]:
    task = st.session_state.get("bg_task")
    if not task:
        return False, "当前没有运行中的任务。"
    proc = task.get("proc")
    if proc is None or proc.poll() is not None:
        return False, "任务已结束，无需停止。"
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return True, "已发送停止信号。"


def _tail_text(path: str, max_lines: int = 300) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _render_task_status(task_name: str, progress_holder, status_holder, log_holder) -> None:
    task = st.session_state.get("bg_task")
    if not task or task.get("name") != task_name:
        return
    proc = task.get("proc")
    running = proc is not None and proc.poll() is None
    elapsed = time.time() - float(task.get("start_ts", time.time()))
    log_text = _tail_text(task.get("log_path", ""), max_lines=400)
    if log_text:
        log_holder.code(log_text, language="text")

    progress_pattern = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")
    m = None
    for ln in reversed(log_text.splitlines()):
        hit = progress_pattern.search(ln)
        if hit:
            m = hit
            break
    if m:
        cur = int(m.group(1))
        total = int(m.group(2))
        p = min(max(cur / max(total, 1), 0.0), 1.0)
        eta_text = ""
        if cur > 0:
            eta_sec = max((elapsed / cur) * (total - cur), 0.0)
            eta_text = f", 预计剩余 {eta_sec:.1f}s"
        progress_holder.progress(p, text=f"进度: {cur}/{total} ({p * 100:.1f}%), 已耗时 {elapsed:.1f}s{eta_text}")
    else:
        progress_holder.progress(0.0, text="任务运行中，等待进度输出...")

    if running:
        status_holder.info(f"运行中，已耗时 {elapsed:.1f}s")
        time.sleep(1.5)
        st.rerun()
    else:
        code = int(proc.returncode or 0) if proc is not None else -1
        if code == 0:
            progress_holder.progress(1.0, text="任务完成")
            status_holder.success(f"任务完成，总耗时 {elapsed:.1f}s")
        else:
            progress_holder.progress(1.0, text="任务失败/已停止")
            status_holder.warning(f"任务结束，退出码={code}，总耗时 {elapsed:.1f}s")


@st.cache_data(ttl=60)
def _get_stock_options(db_path: str) -> pd.DataFrame:
    with _connect(db_path) as conn:
        has_daily = _table_exists(conn, "daily_prices")
        kschema = _kline_schema(conn)
        has_kline = kschema in ("new", "legacy")
        has_info = _table_exists(conn, "stock_info")
        if not has_daily and not has_kline:
            return pd.DataFrame(columns=["ts_code", "name", "industry", "market", "symbol", "list_date"])
        # Prefer kline_data (crawler writes here). Fallback to daily_prices.
        if kschema == "new":
            base = pd.read_sql_query("SELECT DISTINCT ts_code FROM kline_data WHERE frequency='daily'", conn)
        elif kschema == "legacy":
            base = pd.read_sql_query("SELECT DISTINCT ts_code FROM kline_data WHERE timeframe='daily'", conn)
        elif has_daily:
            base = pd.read_sql_query("SELECT DISTINCT ts_code FROM daily_prices", conn)
        else:
            return pd.DataFrame(columns=["ts_code", "name", "industry", "market", "symbol", "list_date"])
        if has_info:
            info = pd.read_sql_query(
                "SELECT ts_code,symbol,name,market,industry,list_date FROM stock_info",
                conn,
            )
            out = base.merge(info, on="ts_code", how="left")
        else:
            out = base.copy()
            for c in ["symbol", "name", "market", "industry", "list_date"]:
                out[c] = None
    out["name"] = out["name"].fillna("")
    out["display"] = out.apply(
        lambda r: f"{r['ts_code']} - {r['name']}" if str(r["name"]).strip() else str(r["ts_code"]),
        axis=1,
    )
    return out.sort_values("ts_code").reset_index(drop=True)


@st.cache_data(ttl=60)
def _get_db_global_range(db_path: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    with _connect(db_path) as conn:
        kschema = _kline_schema(conn)
        # Prefer kline_data (crawler table). Fallback to daily_prices.
        if kschema == "new":
            row = conn.execute(
                "SELECT MIN(substr(datetime,1,10)), MAX(substr(datetime,1,10)), COUNT(*) FROM kline_data WHERE frequency='daily'"
            ).fetchone()
            if row and row[0] and row[1]:
                return pd.to_datetime(row[0]), pd.to_datetime(row[1]), int(row[2] or 0)
        if kschema == "legacy":
            row = conn.execute(
                "SELECT MIN(substr(trade_time,1,10)), MAX(substr(trade_time,1,10)), COUNT(*) FROM kline_data WHERE timeframe='daily'"
            ).fetchone()
            if row and row[0] and row[1]:
                return pd.to_datetime(row[0]), pd.to_datetime(row[1]), int(row[2] or 0)
        if _table_exists(conn, "daily_prices"):
            row = conn.execute("SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM daily_prices").fetchone()
            if row and row[0] and row[1]:
                return pd.to_datetime(row[0]), pd.to_datetime(row[1]), int(row[2] or 0)
    return None, None, 0


@st.cache_data(ttl=60)
def _get_symbol_date_range(db_path: str, ts_code: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    with _connect(db_path) as conn:
        kschema = _kline_schema(conn)
        # Prefer kline_data (crawler table). Fallback to daily_prices.
        if kschema == "new":
            row = conn.execute(
                "SELECT MIN(substr(datetime,1,10)), MAX(substr(datetime,1,10)), COUNT(*) FROM kline_data WHERE ts_code=? AND frequency='daily'",
                (ts_code,),
            ).fetchone()
            if row and row[0] and row[1]:
                return pd.to_datetime(row[0]), pd.to_datetime(row[1]), int(row[2] or 0)
        if kschema == "legacy":
            row = conn.execute(
                "SELECT MIN(substr(trade_time,1,10)), MAX(substr(trade_time,1,10)), COUNT(*) FROM kline_data WHERE ts_code=? AND timeframe='daily'",
                (ts_code,),
            ).fetchone()
            if row and row[0] and row[1]:
                return pd.to_datetime(row[0]), pd.to_datetime(row[1]), int(row[2] or 0)
        if _table_exists(conn, "daily_prices"):
            row = conn.execute(
                "SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM daily_prices WHERE ts_code=?",
                (ts_code,),
            ).fetchone()
            if row and row[0] and row[1]:
                return pd.to_datetime(row[0]), pd.to_datetime(row[1]), int(row[2] or 0)
    return None, None, 0


@st.cache_data(ttl=20)
def _get_kline_daily(db_path: str, ts_code: str, start_date: str, end_date: str, max_rows: int = 5000) -> pd.DataFrame:
    with _connect(db_path) as conn:
        kschema = _kline_schema(conn)
        # Prefer kline_data (crawler table). Fallback to daily_prices.
        if kschema == "new":
            df = pd.read_sql_query(
                """
                SELECT ts_code, substr(datetime,1,10) AS datetime, open, high, low, close, volume, amount
                FROM kline_data
                WHERE ts_code=? AND frequency='daily' AND substr(datetime,1,10)>=? AND substr(datetime,1,10)<=?
                ORDER BY datetime ASC
                LIMIT ?
                """,
                conn,
                params=(ts_code, start_date, end_date, max_rows),
            )
        elif kschema == "legacy":
            df = pd.read_sql_query(
                """
                SELECT ts_code, substr(trade_time,1,10) AS datetime, open, high, low, close, volume, NULL AS amount
                FROM kline_data
                WHERE ts_code=? AND timeframe='daily' AND substr(trade_time,1,10)>=? AND substr(trade_time,1,10)<=?
                ORDER BY datetime ASC
                LIMIT ?
                """,
                conn,
                params=(ts_code, start_date, end_date, max_rows),
            )
        elif _table_exists(conn, "daily_prices"):
            df = pd.read_sql_query(
                """
                SELECT ts_code, trade_date AS datetime, open, high, low, close, volume, amount
                FROM daily_prices
                WHERE ts_code=? AND trade_date>=? AND trade_date<=?
                ORDER BY trade_date ASC
                LIMIT ?
                """,
                conn,
                params=(ts_code, start_date, end_date, max_rows),
            )
        else:
            return pd.DataFrame()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)


def _build_candle_volume_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#d32f2f",
            decreasing_line_color="#2e7d32",
        ),
        row=1,
        col=1,
    )
    colors = ["#d32f2f" if c >= o else "#2e7d32" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df["datetime"], y=df["volume"], marker_color=colors, name="Volume"), row=2, col=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=760,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    return fig


def _train_tab() -> None:
    st.subheader("模型训练")
    st.caption("后台任务模式，支持实时日志与停止任务。")

    db_files = _list_data_db_files()
    db_opts = [str(p) for p in db_files]
    if not db_opts:
        st.warning("No .db files found.")
        return
    db_idx, _ = _default_db_choice(db_opts)

    train_mode = st.radio("训练数据来源", ["raw数据库（现算特征）", "engineered数据库（直接训练）"], horizontal=True, key="train_mode")
    db_path = st.selectbox("训练数据库", db_opts, index=db_idx, key="train_db")

    use_full = st.checkbox("使用全市场（SAMPLE_STOCKS=None）", value=(config.SAMPLE_STOCKS is None), key="train_use_full")
    sample_n = None if use_full else st.number_input("SAMPLE_STOCKS", min_value=10, max_value=10000, value=300, step=10, key="train_sample")

    csplit1, csplit2, csplit3 = st.columns(3)
    with csplit1:
        train_start = st.text_input("Train Start", value=config.TRAIN_START, key="train_start")
        train_end = st.text_input("Train End", value=config.TRAIN_END, key="train_end")
    with csplit2:
        valid_start = st.text_input("Valid Start", value=config.VALID_START, key="train_valid_start")
        valid_end = st.text_input("Valid End", value=config.VALID_END, key="train_valid_end")
    with csplit3:
        test_start = st.text_input("Test Start", value=config.TEST_START, key="train_test_start")
        test_end = st.text_input("Test End", value=config.TEST_END, key="train_test_end")

    p1, p2, p3 = st.columns(3)
    with p1:
        n_estimators = st.number_input("n_estimators", min_value=100, max_value=5000, value=1000, step=100, key="train_n_estimators")
        learning_rate = st.number_input("learning_rate", min_value=0.001, max_value=0.5, value=0.03, step=0.005, format="%.3f", key="train_learning_rate")
        num_leaves = st.number_input("num_leaves", min_value=8, max_value=512, value=64, step=8, key="train_num_leaves")
    with p2:
        max_depth = st.number_input("max_depth", min_value=-1, max_value=20, value=-1, step=1, key="train_max_depth")
        min_child_samples = st.number_input("min_child_samples", min_value=1, max_value=1000, value=50, step=5, key="train_min_child_samples")
        subsample = st.number_input("subsample", min_value=0.1, max_value=1.0, value=0.8, step=0.05, format="%.2f", key="train_subsample")
    with p3:
        colsample_bytree = st.number_input("colsample_bytree", min_value=0.1, max_value=1.0, value=0.8, step=0.05, format="%.2f", key="train_colsample_bytree")
        reg_alpha = st.number_input("reg_alpha", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="train_reg_alpha")
        reg_lambda = st.number_input("reg_lambda", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="train_reg_lambda")
        topk = st.number_input("TopK", min_value=1, max_value=200, value=config.TOPK, step=1, key="train_topk")
        max_per_sector = st.number_input("max_per_sector", min_value=1, max_value=20, value=config.MAX_PER_SECTOR, step=1, key="train_max_per_sector")

    b1, b2 = st.columns(2)
    run_btn = b1.button("开始训练", type="primary", use_container_width=True, key="btn_train_run")
    stop_btn = b2.button("停止训练", use_container_width=True, key="btn_train_stop")

    progress_holder = st.empty()
    status_holder = st.empty()
    log_holder = st.empty()

    if stop_btn:
        ok, msg = _stop_bg_task()
        if ok:
            status_holder.warning(msg)
        else:
            status_holder.info(msg)

    if run_btn:
        try:
            jobs_dir = Path(__file__).resolve().parent / "jobs"
            cmd = [
                sys.executable,
                str(jobs_dir / "train_job.py"),
                "--db-path", str(Path(db_path).resolve()),
                "--mode", "raw" if train_mode.startswith("raw数据库") else "engineered",
                "--eng-table", ENG_TABLE,
                "--sample-stocks", "0" if use_full else str(int(sample_n)),
                "--train-start", train_start,
                "--train-end", train_end,
                "--valid-start", valid_start,
                "--valid-end", valid_end,
                "--test-start", test_start,
                "--test-end", test_end,
                "--topk", str(int(topk)),
                "--max-per-sector", str(int(max_per_sector)),
                "--n-estimators", str(int(n_estimators)),
                "--learning-rate", str(float(learning_rate)),
                "--num-leaves", str(int(num_leaves)),
                "--max-depth", str(int(max_depth)),
                "--min-child-samples", str(int(min_child_samples)),
                "--subsample", str(float(subsample)),
                "--colsample-bytree", str(float(colsample_bytree)),
                "--reg-alpha", str(float(reg_alpha)),
                "--reg-lambda", str(float(reg_lambda)),
            ]
            _start_bg_task("train", cmd, cwd=Path(__file__).resolve().parent)
            status_holder.info("训练任务已启动。")
        except Exception as e:
            status_holder.error(str(e))

    _render_task_status("train", progress_holder, status_holder, log_holder)

    out_dir = Path(config.OUTPUT_DIR)
    chart_options = {
        "累计收益曲线": out_dir / "cumulative_return.png",
        "特征重要性 Top30": out_dir / "feature_importance_top30.png",
        "分数分布": out_dir / "score_distribution.png",
        "分组收益曲线(Q1~Q5)": out_dir / "quintile_cumulative.png",
        "年度收益柱状图": out_dir / "yearly_returns.png",
    }
    existing = {k: v for k, v in chart_options.items() if v.exists()}
    if existing:
        st.markdown("---")
        chosen = st.selectbox("训练图表", list(existing.keys()), key="train_chart_pick")
        st.image(str(existing[chosen]), caption=chosen, use_container_width=True)

def _feature_engineering_tab() -> None:
    st.subheader("特征工程")
    st.caption("后台任务模式，支持实时日志与停止任务。")

    bundles = _list_model_bundles()
    if not bundles:
        st.warning("未找到可用模型与特征清单，请先训练模型。")
        return
    bundle_labels = [b["label"] for b in bundles]
    selected_bundle_label = st.selectbox("模型组合", bundle_labels, key="fe_model_bundle")
    bundle = next(b for b in bundles if b["label"] == selected_bundle_label)

    db_files = _list_data_db_files()
    db_opts = [str(p) for p in db_files]
    db_idx, _ = _default_db_choice(db_opts)
    raw_db = st.selectbox("原始数据库", db_opts, index=db_idx, key="fe_raw_db")
    out_db_default = f"../data/engineered_{Path(bundle['model_path']).stem}.db"
    out_db = st.text_input("输出 engineered 数据库路径", value=out_db_default, key="fe_out_db")

    use_full = st.checkbox("使用全市场（SAMPLE_STOCKS=None）", value=True, key="fe_full")
    sample_n = None if use_full else st.number_input("SAMPLE_STOCKS", min_value=10, max_value=10000, value=300, step=10, key="fe_sample")

    b1, b2 = st.columns(2)
    run_btn = b1.button("执行特征工程", type="primary", use_container_width=True, key="btn_fe_run")
    stop_btn = b2.button("停止特征工程", use_container_width=True, key="btn_fe_stop")

    progress_holder = st.empty()
    status_holder = st.empty()
    log_holder = st.empty()

    if stop_btn:
        ok, msg = _stop_bg_task()
        if ok:
            status_holder.warning(msg)
        else:
            status_holder.info(msg)

    if run_btn:
        try:
            jobs_dir = Path(__file__).resolve().parent / "jobs"
            cmd = [
                sys.executable,
                str(jobs_dir / "feature_job.py"),
                "--raw-db", str(Path(raw_db).resolve()),
                "--out-db", str(Path(out_db).resolve()),
                "--feature-csv", str(Path(bundle["feature_path"]).resolve()),
                "--eng-table", ENG_TABLE,
                "--sample-stocks", "0" if use_full else str(int(sample_n)),
            ]
            _start_bg_task("feature", cmd, cwd=Path(__file__).resolve().parent)
            status_holder.info("特征工程任务已启动。")
        except Exception as e:
            status_holder.error(str(e))

    _render_task_status("feature", progress_holder, status_holder, log_holder)

def _predict_tab() -> None:
    st.subheader("模型预测")
    st.caption("从 models 选择模型，支持 raw/engineered 数据库。")

    bundles = _list_model_bundles()
    if not bundles:
        st.warning("models 目录未找到可用模型。")
        return
    bundle_labels = [b["label"] for b in bundles]
    selected_bundle_label = st.selectbox("选择模型", bundle_labels, key="pred_bundle")
    bundle = next(b for b in bundles if b["label"] == selected_bundle_label)
    required_features = _read_feature_list(bundle["feature_path"])

    db_mode = st.radio("数据库类型", ["engineered数据库", "raw数据库"], horizontal=True, key="pred_db_mode")
    db_files = _list_data_db_files()
    db_opts = [str(p) for p in db_files]
    db_idx, _ = _default_db_choice(db_opts)
    db_path = st.selectbox("选择数据库", db_opts, index=db_idx, key="pred_db")

    # Quick metadata preview without forcing full feature build.
    try:
        gmin, gmax, grow = _get_db_global_range(db_path)
        if gmin is not None:
            st.info(f"数据库范围: {gmin.date()} -> {gmax.date()}，总行数: {grow:,}")
    except Exception:
        pass

    target_date = st.text_input("预测日期 (YYYY-MM-DD, 留空=最近交易日)", value="", key="pred_target_date")
    topk = st.number_input("TopK", min_value=1, max_value=200, value=config.TOPK, step=1, key="pred_topk")
    max_per_sector = st.number_input("max_per_sector", min_value=1, max_value=20, value=config.MAX_PER_SECTOR, step=1, key="pred_max_per_sector")

    run_btn = st.button("执行预测", type="primary", use_container_width=True, key="btn_pred_run")
    pred_progress = st.empty()
    pred_status = st.empty()
    if run_btn:
        t_pred = time.time()
        step = 0
        total = 4

        def _pred_step(msg: str) -> None:
            nonlocal step
            step += 1
            p = min(step / total, 1.0)
            pred_progress.progress(p, text=f"{msg} | 已耗时 {time.time() - t_pred:.1f}s")

        try:
            _pred_step("加载数据")
            if db_mode.startswith("engineered"):
                feat_df = _load_engineered_df(db_path, table_name=ENG_TABLE)
                ok, miss = _validate_feature_requirements(feat_df, required_features)
                if not ok:
                    st.error(f"engineered 数据库缺少模型所需特征: {miss[:20]}")
                    return
            else:
                feat_df, _ = _prepare_feature_df_from_raw(db_path, sample_stocks=None)
                ok, miss = _validate_feature_requirements(feat_df, required_features)
                if not ok:
                    st.error(f"raw 数据构建后仍缺少模型所需特征: {miss[:20]}")
                    return

            _pred_step("准备预测日期")
            dates = sorted(pd.to_datetime(feat_df["date"]).dropna().unique())
            if not dates:
                st.error("没有可用于预测的日期。")
                return
            latest = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
            use_date = target_date.strip() or latest

            _pred_step("执行预测与排序")
            res = predict_topk_for_date_with_saved_model(
                feature_df=feat_df,
                feature_cols=required_features,
                target_date=use_date,
                model_path=bundle["model_path"],
                topk=int(topk),
                max_per_sector=int(max_per_sector),
            )
            out = res.topk.copy()
            out.insert(0, "rank", range(1, len(out) + 1))
            _pred_step("输出结果")
            st.success(f"预测完成，日期: {use_date}，返回 {len(out)} 条。")
            pred_status.info(f"预测总耗时 {time.time() - t_pred:.1f}s")
            st.dataframe(out, use_container_width=True, height=460)
            csv = out.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="下载推荐结果 CSV",
                data=csv.encode("utf-8-sig"),
                file_name=f"pit_topk_{pd.Timestamp(use_date).strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        except Exception as e:
            pred_progress.progress(1.0, text=f"预测失败 | 已耗时 {time.time() - t_pred:.1f}s")
            st.error(f"预测失败: {e}")


def _visual_tab() -> None:
    st.subheader("数据库可视化")
    st.caption("支持代码/名称搜索、区间快捷选择和自定义日期。")

    db_opts = [str(p) for p in _list_data_db_files()]
    if not db_opts:
        st.warning("未找到数据库文件。")
        return
    db_idx, _ = _default_db_choice(db_opts)

    selected_db = st.selectbox("数据库", db_opts, index=db_idx, key="viz_db_select")
    db_path_input = st.text_input("数据库路径（可手动覆盖）", value=selected_db, key="viz_db_path")
    if st.button("加载数据库", key="viz_load_btn", type="primary"):
        st.session_state["viz_active_db"] = db_path_input
        _get_stock_options.clear()
        _get_db_global_range.clear()
        _get_symbol_date_range.clear()
        _get_kline_daily.clear()

    db_path = st.session_state.get("viz_active_db", db_path_input)
    if not Path(db_path).exists():
        st.error(f"数据库不存在: {db_path}")
        return
    gmin, gmax, grow = _get_db_global_range(db_path)
    if gmin is not None:
        st.info(f"当前库全局范围: {gmin.date()} -> {gmax.date()}，总行数: {grow:,}")

    stock_df = _get_stock_options(db_path)
    if stock_df.empty:
        st.warning("未找到可视化股票数据。")
        return

    search = st.text_input("搜索股票（代码或名称）", value="", key="viz_search")
    filtered = stock_df
    if search.strip():
        kw = search.strip().lower()
        filtered = stock_df[
            stock_df["ts_code"].str.lower().str.contains(kw, na=False)
            | stock_df["name"].astype(str).str.lower().str.contains(kw, na=False)
        ].copy()
    if filtered.empty:
        st.warning("没有匹配股票。")
        return

    display = st.selectbox("股票（代码 + 中文名）", filtered["display"].tolist(), index=0, key="viz_stock_select")
    sel = filtered[filtered["display"] == display].iloc[0]
    ts_code = str(sel["ts_code"])

    min_dt, max_dt, row_cnt = _get_symbol_date_range(db_path, ts_code)
    if min_dt is None:
        st.warning("该股票无日线数据。")
        return
    cover_years = (max_dt - min_dt).days / 365.25
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("股票代码", ts_code)
    m2.metric("股票名称", str(sel.get("name") or "-"))
    m3.metric("行业", str(sel.get("industry") or "-"))
    m4.metric("市场", str(sel.get("market") or "-"))
    m5.metric("数据起始", str(min_dt.date()))
    m6.metric("该股票可用行数", f"{row_cnt:,}")
    c1, c2 = st.columns(2)
    c1.metric("数据结束", str(max_dt.date()))
    c2.metric("覆盖年数(约)", f"{cover_years:.2f}")

    preset = st.radio("区间快捷", ["7日", "30日", "90日", "1年", "5年", "自定义"], horizontal=True, key="viz_preset")
    if preset == "自定义":
        start_date, end_date = st.date_input(
            "自定义区间",
            value=(max(min_dt.date(), max_dt.date() - timedelta(days=90)), max_dt.date()),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key="viz_date_range",
        )
        if isinstance(start_date, tuple):
            start_date, end_date = start_date
    else:
        days = {"7日": 7, "30日": 30, "90日": 90, "1年": 365, "5年": 365 * 5}[preset]
        end_date = max_dt.date()
        start_date = max(min_dt.date(), end_date - timedelta(days=days))
    max_rows = st.slider("最大读取行数", min_value=200, max_value=10000, value=3000, step=100, key="viz_max_rows")

    df = _get_kline_daily(db_path, ts_code, str(start_date), str(end_date), max_rows=max_rows)
    if df.empty:
        st.warning("该区间无数据。")
        return
    st.plotly_chart(_build_candle_volume_figure(df), use_container_width=True)
    with st.expander("原始数据预览"):
        show = df.copy()
        show["datetime"] = show["datetime"].dt.strftime("%Y-%m-%d")
        st.dataframe(show.iloc[::-1], use_container_width=True, height=320)


def _run_ingestion_cmd(cmd: list[str], log_holder, progress_holder, status_holder) -> int:
    """
    在UI中执行抓取命令并实时展示日志与进度。
    """
    lines: list[str] = []
    progress_holder.progress(0.0, text="任务已启动，等待抓取器输出进度...")
    progress_pattern = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        log_holder.code("\n".join(lines[-500:]), language="text")
        status_holder.info(f"运行中: {line[-120:]}" if line else "运行中...")

        m = progress_pattern.search(line)
        if m:
            cur = int(m.group(1))
            total = int(m.group(2))
            if total > 0:
                p = min(max(cur / total, 0.0), 1.0)
                progress_holder.progress(p, text=f"抓取进度: {cur}/{total} ({p * 100:.1f}%)")

    proc.wait()
    if int(proc.returncode or 0) == 0:
        progress_holder.progress(1.0, text="抓取完成")
    else:
        progress_holder.progress(1.0, text="抓取失败")
    return int(proc.returncode or 0)


def _ingestion_tab() -> None:
    st.subheader("数据抓取")
    st.caption("按时间范围 + 股票范围抓取数据并写入目标数据库（增量模式，避免重复写入）。")

    ingest_script = Path(__file__).resolve().parent.parent / "app" / "ingestion" / "run_ingestion.py"
    if not ingest_script.exists():
        st.error(f"未找到统一抓取入口: {ingest_script}")
        return

    db_opts = [str(p) for p in _list_data_db_files()]
    if not db_opts:
        st.warning("未找到可写入的数据库文件。")
        return
    db_idx, _ = _default_db_choice(db_opts)

    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        db_path = st.selectbox("目标数据库", db_opts, index=db_idx, key="ing_db_path")
        freq = st.selectbox("抓取频率", ["daily", "1min", "5min", "60min", "1min,5min,60min,daily"], index=0, key="ing_freq")
        d1, d2 = st.columns(2)
        min_pick_date = date(1990, 1, 1)
        max_pick_date = datetime.now().date()
        with d1:
            date_start = st.date_input(
                "开始日期",
                value=datetime(1998, 1, 1).date(),
                min_value=min_pick_date,
                max_value=max_pick_date,
                key="ing_start_date",
            )
        with d2:
            date_end = st.date_input(
                "截止日期",
                value=datetime.now().date(),
                min_value=min_pick_date,
                max_value=max_pick_date,
                key="ing_end_date",
            )
        o1, o2 = st.columns(2)
        with o1:
            connect_timeout = st.number_input(
                "连接超时(秒)",
                min_value=1.0,
                max_value=60.0,
                value=8.0,
                step=1.0,
                key="ing_connect_timeout",
            )
        with o2:
            read_timeout = st.number_input(
                "读取超时(秒)",
                min_value=5.0,
                max_value=180.0,
                value=30.0,
                step=5.0,
                key="ing_read_timeout",
            )
        o3, o4 = st.columns(2)
        with o3:
            sleep_min = st.number_input(
                "最小间隔(秒)",
                min_value=0.05,
                max_value=5.0,
                value=0.20,
                step=0.05,
                key="ing_sleep_min",
            )
        with o4:
            sleep_max = st.number_input(
                "最大间隔(秒)",
                min_value=0.05,
                max_value=5.0,
                value=0.50,
                step=0.05,
                key="ing_sleep_max",
            )
        max_retries = st.number_input(
            "重试次数",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="ing_max_retries",
        )

    with c2:
        stock_mode = st.radio("股票范围", ["单只股票", "多只股票", "全市场"], horizontal=True, key="ing_stock_mode")
        skip_features = st.checkbox(
            "抓取后跳过特征回写（推荐）",
            value=True,
            key="ing_skip_features",
        )
        refresh_stock_info = st.checkbox(
            "本次同步更新 stock_info",
            value=False,
            key="ing_refresh_stock_info",
        )
        stock_df = _get_stock_options(db_path)
        symbol = None
        symbols_text = ""

        if stock_mode == "单只股票":
            if stock_df.empty:
                st.warning("当前数据库无 stock_info，可手动输入股票代码。")
                symbol = st.text_input("股票代码（如 000001.SZ）", value="", key="ing_single_manual").strip().upper()
            else:
                pick = st.selectbox("选择股票（代码 + 中文名）", stock_df["display"].tolist(), key="ing_single_pick")
                symbol = str(stock_df.loc[stock_df["display"] == pick, "ts_code"].iloc[0]).upper()
        elif stock_mode == "多只股票":
            if stock_df.empty:
                symbols_text = st.text_area("输入股票代码（逗号分隔）", value="000001.SZ,600000.SH", key="ing_multi_manual")
            else:
                picks = st.multiselect(
                    "选择多只股票（代码 + 中文名）",
                    options=stock_df["display"].tolist(),
                    default=stock_df["display"].tolist()[:2],
                    key="ing_multi_pick",
                )
                ts_list = []
                for d in picks:
                    ts_list.append(str(stock_df.loc[stock_df["display"] == d, "ts_code"].iloc[0]).upper())
                symbols_text = ",".join(ts_list)

        st.info("说明：抓取命令默认使用 --incremental，数据库已有数据会自动跳过（避免重复）。")

    b1, b2 = st.columns(2)
    run_btn = b1.button("开始抓取", type="primary", use_container_width=True, key="btn_ingest_run")
    stop_btn = b2.button("停止抓取", use_container_width=True, key="btn_ingest_stop")
    progress_holder = st.empty()
    log_holder = st.empty()
    status_holder = st.empty()

    if stop_btn:
        ok, msg = _stop_bg_task()
        if ok:
            status_holder.warning(msg)
        else:
            status_holder.info(msg)

    if run_btn:
        if not Path(db_path).exists():
            st.error(f"数据库不存在: {db_path}")
            return
        if date_start > date_end:
            st.error("开始日期不能晚于结束日期。")
            return

        start_yyyymmdd = pd.Timestamp(date_start).strftime("%Y%m%d")
        end_yyyymmdd = pd.Timestamp(date_end).strftime("%Y%m%d")

        cmd = [
            sys.executable,
            str(ingest_script),
            "kline",
            "--db-path",
            str(Path(db_path).resolve()),
            "--frequencies",
            freq,
            "--start-date",
            start_yyyymmdd,
            "--end-date",
            end_yyyymmdd,
            "--incremental",
            "--progress-every",
            "20",
            "--connect-timeout",
            str(connect_timeout),
            "--read-timeout",
            str(read_timeout),
            "--sleep-min",
            str(sleep_min),
            "--sleep-max",
            str(sleep_max),
            "--max-retries",
            str(int(max_retries)),
        ]
        if skip_features:
            cmd += ["--skip-features"]
        if refresh_stock_info:
            cmd += ["--refresh-stock-info"]

        if stock_mode == "单只股票":
            if not symbol:
                st.error("请选择或输入一只股票。")
                return
            cmd += ["--single", symbol]
        elif stock_mode == "多只股票":
            symbols_text = symbols_text.strip()
            if not symbols_text:
                st.error("请至少选择一只股票。")
                return
            cmd += ["--symbols", symbols_text]
        else:
            cmd += ["--full-market"]

        try:
            _start_bg_task("ingest", cmd, cwd=Path(__file__).resolve().parent)
            status_holder.info("抓取任务已启动。")
        except Exception as e:
            status_holder.error(str(e))

    _render_task_status("ingest", progress_holder, status_holder, log_holder)
    task = st.session_state.get("bg_task")
    if task and task.get("name") == "ingest":
        proc = task.get("proc")
        if proc is not None and proc.poll() == 0:
            _get_stock_options.clear()
            _get_db_global_range.clear()
            _get_symbol_date_range.clear()
            _get_kline_daily.clear()


def main() -> None:
    st.title("A-Share ML Control Panel")
    tab_ing, tab_viz, tab_fe, tab_train, tab_predict = st.tabs(
        ["数据抓取", "数据可视化", "特征工程", "模型训练", "模型预测"]
    )
    with tab_ing:
        _ingestion_tab()
    with tab_viz:
        _visual_tab()
    with tab_fe:
        _feature_engineering_tab()
    with tab_train:
        _train_tab()
    with tab_predict:
        _predict_tab()


if __name__ == "__main__":
    main()
