import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="A-Share Kline Dashboard", layout="wide")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg-start: #f4f7f2;
          --bg-end: #e6eef3;
          --card: #ffffff;
          --ink: #1f2d3d;
          --muted: #5f6b7a;
          --up: #d32f2f;
          --down: #2e7d32;
        }
        .stApp {
          background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
          color: var(--ink);
        }
        .block-container {
          padding-top: 1rem;
          padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30)
def list_symbols(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT ts_code FROM kline_data ORDER BY ts_code").fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=30)
def list_frequencies(db_path: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT frequency FROM kline_data ORDER BY frequency").fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=10)
def load_data(
    db_path: str,
    ts_code: str,
    frequency: str,
    start_dt: str,
    end_dt: str,
    limit_rows: int,
) -> pd.DataFrame:
    sql = """
    SELECT ts_code, datetime, open, high, low, close, volume, amount, frequency
    FROM kline_data
    WHERE ts_code = ? AND frequency = ? AND datetime >= ? AND datetime <= ?
    ORDER BY datetime DESC
    LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=(ts_code, frequency, start_dt, end_dt, limit_rows))
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df


def candlestick_chart(df: pd.DataFrame) -> alt.Chart:
    up_down = df.copy()
    up_down["dir"] = up_down.apply(lambda x: "up" if x["close"] >= x["open"] else "down", axis=1)

    wick = alt.Chart(up_down).mark_rule().encode(
        x=alt.X("datetime:T", title="Time"),
        y=alt.Y("low:Q", title="Price"),
        y2="high:Q",
        color=alt.Color("dir:N", scale=alt.Scale(domain=["up", "down"], range=["#d32f2f", "#2e7d32"]), legend=None),
    )

    body = alt.Chart(up_down).mark_bar(size=6).encode(
        x="datetime:T",
        y="open:Q",
        y2="close:Q",
        color=alt.Color("dir:N", scale=alt.Scale(domain=["up", "down"], range=["#d32f2f", "#2e7d32"]), legend=None),
        tooltip=["ts_code:N", "datetime:T", "open:Q", "high:Q", "low:Q", "close:Q", "volume:Q", "amount:Q"],
    )

    return (wick + body).properties(height=420)


def volume_chart(df: pd.DataFrame) -> alt.Chart:
    tmp = df.copy()
    tmp["dir"] = tmp.apply(lambda x: "up" if x["close"] >= x["open"] else "down", axis=1)
    return (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x="datetime:T",
            y=alt.Y("volume:Q", title="Volume"),
            color=alt.Color("dir:N", scale=alt.Scale(domain=["up", "down"], range=["#d32f2f", "#2e7d32"]), legend=None),
            tooltip=["datetime:T", "volume:Q", "amount:Q"],
        )
        .properties(height=180)
    )


def main() -> None:
    inject_style()
    st.title("Eastmoney Crawler Data Dashboard")

    default_db = "market_kline.db" if Path("market_kline.db").exists() else "market.db"
    db_path = st.sidebar.text_input("SQLite DB Path", value=default_db)
    if not Path(db_path).exists():
        st.error(f"DB not found: {db_path}")
        st.stop()

    try:
        symbols = list_symbols(db_path)
        frequencies = list_frequencies(db_path)
    except Exception as exc:
        st.error(f"Failed to open DB: {exc}")
        st.stop()

    if not symbols or not frequencies:
        st.warning("No data in kline_data table.")
        st.stop()

    ts_code = st.sidebar.selectbox("Symbol", symbols, index=0)
    frequency = st.sidebar.selectbox("Frequency", frequencies, index=0)

    end_date = date.today()
    start_date = end_date - timedelta(days=45)
    d1, d2 = st.sidebar.date_input("Date Range", value=(start_date, end_date))
    limit_rows = st.sidebar.slider("Max Rows", min_value=300, max_value=20000, value=5000, step=100)

    start_dt = f"{d1} 00:00:00"
    end_dt = f"{d2} 23:59:59"

    df = load_data(
        db_path=db_path,
        ts_code=ts_code,
        frequency=frequency,
        start_dt=start_dt,
        end_dt=end_dt,
        limit_rows=limit_rows,
    )

    if df.empty:
        st.warning("No records for selected filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Latest", df["datetime"].max().strftime("%Y-%m-%d %H:%M"))
    c3.metric("Close", f"{df['close'].iloc[-1]:.2f}")
    c4.metric("Range", f"{df['datetime'].min().strftime('%Y-%m-%d')} -> {df['datetime'].max().strftime('%Y-%m-%d')}")

    st.altair_chart(candlestick_chart(df), use_container_width=True)
    st.altair_chart(volume_chart(df), use_container_width=True)

    with st.expander("Raw Data"):
        view = df.copy()
        view["datetime"] = view["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(view.iloc[::-1], use_container_width=True, height=320)


if __name__ == "__main__":
    main()
