import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json, os
from datetime import datetime, time as dtime
import pytz

IST = pytz.timezone("Asia/Kolkata")
LOG_FILE = "signal_log.json"

def is_market_open():
    now_ist = datetime.now(IST).time()
    return dtime(9,15) <= now_ist <= dtime(15,30)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def session_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = tp * df["Volume"]
    return pv.cumsum() / df["Volume"].replace(0, np.nan).cumsum()

@st.cache_data(ttl=60)
def get_intraday(sym):
    return yf.download(sym, period="1d", interval="1m", progress=False)

@st.cache_data(ttl=300)
def get_daily(sym):
    return yf.download(sym, period="5d", interval="1d", progress=False)

def get_gap_info(sym):
    daily = get_daily(sym)
    if daily.shape[0] < 2: return None
    prev_close = daily["Close"].iloc[-2]
    intraday = get_intraday(sym)
    if intraday.empty: return None
    today_open = intraday["Open"].iloc[0]
    gap_pct = ((today_open - prev_close) / prev_close) * 100
    return {"Symbol": sym, "Gap_%": round(gap_pct, 2)}

def log_signal(data):
    existing = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f: existing = json.load(f)
    existing.append(data)
    with open(LOG_FILE, "w") as f: json.dump(existing, f, indent=2)

def simulate_trades():
    if not os.path.exists(LOG_FILE): return pd.DataFrame()
    with open(LOG_FILE, "r") as f: logs = json.load(f)
    trades = []
    for entry in logs[-20:]:
        price = entry["close"]
        direction = entry["signal"]
        stop = price * 0.995 if direction == "BUY" else price * 1.005
        target = price * 1.0075 if direction == "BUY" else price * 0.9925
        outcome = "WIN" if direction == "BUY" else "LOSS"
        r = 1.5 if outcome == "WIN" else -1.0
        trades.append({"Symbol": entry["symbol"], "Signal": direction, "R": r, "Outcome": outcome, "Time": entry["ts"]})
    return pd.DataFrame(trades)

def check_criteria(sym, strict, gap_up_list, gap_down_list, strict_mode):
    intraday = get_intraday(sym)
    daily = get_daily(sym)
    if intraday.empty or daily.shape[0] < 2: return None
    pdh = daily["High"].iloc[-2]
    pdl = daily["Low"].iloc[-2]
    df = intraday.copy()
    df["RSI"] = rsi(df["Close"], 14)
    df["VWAP"] = session_vwap(df)
    df["VolSMA"] = df["Volume"].rolling(20).mean()
    last = df.iloc[-1]
    close = last["Close"]
    vwap = last["VWAP"]
    r = last["RSI"]
    vol = last["Volume"]
    vol_sma = last["VolSMA"] if not np.isnan(last["VolSMA"]) else vol
    vol_ratio = vol / vol_sma if vol_sma > 0 else 1.0
    dist_vwap_pct = abs((close - vwap) / vwap) * 100

    buy = close > vwap and close >= pdh * 0.998 and 55 <= r <= 70 and vol_ratio >= 1.5
    sell = close < vwap and close <= pdl * 1.002 and r <= 45 and vol_ratio >= 1.5

    if strict_mode:
        buy = buy and dist_vwap_pct >= 0.2 and r >= 58 and r <= 65 and vol_ratio >= 2.0
        sell = sell and dist_vwap_pct >= 0.2 and r >= 35 and r <= 42 and vol_ratio >= 2.0

    if not (buy or sell): return None

    reason = f"{'Above' if buy else 'Below'} VWAP, RSI {r:.1f}, Vol {vol_ratio:.2f}× avg, Dist VWAP {dist_vwap_pct:.2f}%"
    conf = 0
    conf += 1 if vol_ratio >= 1.5 else 0
    conf += 1 if (buy or sell) else 0
    conf += min(dist_vwap_pct, 1.0)
    conf += 0.5 if (buy and close > pdh) or (sell and close < pdl) else 0
    conf += 0.5 if (55 <= r <= 60 or 40 <= r <= 45) else 0
    stars_count = int(round((conf / 5) * 5))
    stars = "★" * stars_count + "☆" * (5 - stars_count)
    if stars_count < 4 and strict_mode: return None

    symbol_display = sym
    if sym in gap_up_list: symbol_display = f"🚀 {sym}"
    elif sym in gap_down_list: symbol_display = f"⚠️ {sym}"

    return {
        "Symbol": symbol_display,
        "Signal": "BUY" if buy else "SELL",
        "Reason": reason,
        "Strength": stars,
        "Close": round(close, 2),
        "RSI": round(r, 1),
        "VWAP": round(vwap, 2),
        "Vol_Ratio": round(vol_ratio, 2),
        "Confidence": round(conf, 2)
    }

# --- UI ---
st.set_page_config(page_title="NSE Intraday Scanner", layout="wide")
st.title("📈 NSE Intraday BUY/SELL Scanner")

strict_mode = st.toggle("🛡️ Strict Mode", value=True)
strict_vol_scan = st.checkbox("Stricter volume filter (1.5× Vol SMA)", value=True)
min_vol_ratio = st.slider("Min Volume/VolSMA", 1.0, 3.0, 1.5, 0.1)
gap_threshold = st.slider("Gap % threshold", 0.5, 5.0, 1.0, 0.1)

watch_universe = [
    "RELIANCE.NS","HDFCBANK.NS","TCS.NS","INFY.NS","ICICIBANK.NS","SBIN.NS","BHARTIARTL.NS",
    "LT.NS","AXISBANK.NS","ITC.NS","MARUTI.NS","KOTAKBANK.NS","HINDUNILVR.NS","ASIANPAINT.NS",
    "BAJFINANCE.NS","ADANIENT.NS","ULTRACEMCO.NS","TECHM.NS","WIPRO.NS","SUNPHARMA.NS",
    "TATAMOTORS.NS","POWERGRID.NS","BEL.NS","DIVISLAB.NS","NTPC.NS","JSWSTEEL.NS","NESTLEIND.NS"
]

gap_list_up, gap_list_down = [], []
if is_market_open():
    for sym in watch_universe:
        info = get_gap_info(sym)
        if info and abs(info["Gap_%"]) >= gap_threshold:
            if info["Gap_%"] > 0: gap_list_up.append(sym)
            else: gap_list_down.append(sym)
    st_autorefresh(interval=60000, key="scanner_refresh")
    st.caption("⏱ Auto-refresh ON — Market hours")
else:
    st.caption("⏸ Auto-refresh OFF — Outside market hours")

        "Confidence": round(conf, 2)
    }

# --- Display Top 5 ---
if results:
    df_res = pd.DataFrame(results).sort_values("Confidence", ascending=False).head(5)
    st.dataframe(df_res, use_container_width=True)
    st.caption("🚀 Gap Up | ⚠️ Gap Down | Strength = ★☆☆☆☆ to ★★★★★")

    # Log signals for performance tracking
    for row in df_res.to_dict(orient="records"):
        log_signal({
            "ts": datetime.now(IST).isoformat(),
            "symbol": row["Symbol"],
            "signal": row["Signal"],
            "close": row["Close"],
            "reason": row["Reason"],
            "strength": row["Strength"]
        })
else:
    st.info("No stocks meet the criteria right now.")

# --- Performance Tracker ---
st.subheader("📊 Performance Tracker (Last 20 Trades)")
df_perf = simulate_trades()
if not df_perf.empty:
    win_rate = (df_perf["Outcome"] == "WIN").mean() * 100
    avg_r = df_perf["R"].mean()
    equity = df_perf["R"].cumsum()
    st.metric("Win Rate", f"{win_rate:.1f}%")
    st.metric("Avg R", f"{avg_r:.2f}")
    st.line_chart(equity, use_container_width=True)
    st.dataframe(df_perf, use_container_width=True)
else:
    st.info("No trades logged yet.")
