from datetime import datetime, timedelta
import logging
import time

import google.generativeai as genai  # 新增 Gemini SDK
import numpy as np
import ollama
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
import requests
import streamlit as st
import yfinance as yf

# =========================================================
# 1. 頁面設定與初始化
# =========================================================
st.set_page_config(
    page_title="全能 AI 量化投資助理",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault(
    "system_context",
    "你是一位專業的台股量化投資分析師。請依據畫面實際載入的數據回答，不得捏造數據；若資料不足，請明確說明。"
)
st.session_state.setdefault("stock_id", "3481")
st.session_state.setdefault("data_cache", {})

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_per_pbr_and_dividend_yield"
# 更換為真實瀏覽器的 User-Agent，避免被政府 API 阻擋
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 15

st.title("📈 全能 AI 量化投資助理")
st.caption("官方基本面資料、歷史股價、技術指標、量價分析、券商分點與本機 Ollama AI")
st.markdown("---")


# =========================================================
# 2. 共用資料工具
# =========================================================
def safe_float(value):
    """安全轉換數字；排除空值及無效數字。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        text = str(value).strip().replace(",", "").replace("%", "")
        if text in ("", "-", "--", "N/A", "None", "null"):
            return None
        number = float(text)
        if not np.isfinite(number):
            return None
        return number
    except (ValueError, TypeError):
        return None


def format_number(value, decimals=2):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def normalize_stock_id(stock_id):
    """乾淨清理股票代碼，只留純數字（例如 3481）。"""
    value = str(stock_id).strip().upper()
    for suffix in (".TW", ".TWO"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
    return value.strip()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_json(url):
    """下載 JSON；失敗時回傳明確的錯誤訊息。"""
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("API 回傳格式不是預期的清單格式")
    return data


def first_valid_number(row, keys):
    for key in keys:
        if key in row:
            value = safe_float(row.get(key))
            if value is not None:
                return value
    return None


def find_row_by_code(rows, code, code_keys):
    for row in rows:
        for key in code_keys:
            if normalize_stock_id(row.get(key, "")) == code:
                return row
    return None


# =========================================================
# 3. 官方基本面資料
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_official_metrics(stock_id):
    """優先查詢證交所上市股票及櫃買中心上櫃股票。"""
    code = normalize_stock_id(stock_id)
    errors = []

    # 上市股票：臺灣證券交易所
    try:
        rows = fetch_json(TWSE_URL)
        row = find_row_by_code(rows, code, ["Code", "公司代號", "SecuritiesCompanyCode"])
        if row:
            return {
                "source": "臺灣證券交易所",
                "name": row.get("Name", row.get("公司名稱", "")),
                "pe": first_valid_number(row, ["PEratio", "PERatio", "本益比"]),
                "pb": first_valid_number(row, ["PBratio", "PBRatio", "股價淨值比"]),
                "yield": first_valid_number(row, ["DividendYield", "殖利率"]),
                "raw_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
    except Exception as exc:
        errors.append(f"TWSE：{exc}")

    # 上櫃股票：櫃買中心
    try:
        rows = fetch_json(TPEX_URL)
        row = find_row_by_code(rows, code, ["SecuritiesCompanyCode", "Code", "公司代號"])
        if row:
            return {
                "source": "證券櫃檯買賣中心",
                "name": row.get("CompanyName", row.get("Name", row.get("公司名稱", ""))),
                "pe": first_valid_number(row, ["PeRatio", "PERatio", "PEratio", "本益比"]),
                "pb": first_valid_number(row, ["PbRatio", "PBRatio", "PBratio", "股價淨值比"]),
                "yield": first_valid_number(row, ["DividendYield", "殖利率"]),
                "raw_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
    except Exception as exc:
        errors.append(f"TPEX：{exc}")

    return {
        "source": "官方資料查詢失敗",
        "name": "",
        "pe": None,
        "pb": None,
        "yield": None,
        "raw_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": "；".join(errors) if errors else "查無此股票代碼",
    }


# =========================================================
# 4. Yahoo Finance 歷史股價與財務資料
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def download_history(stock_id, start_date, end_date):
    """依序測試上市與上櫃代碼。"""
    code = normalize_stock_id(stock_id)
    if stock_id.startswith("^") or "=" in stock_id:
        tickers_to_try = [stock_id]
    else:
        tickers_to_try = [f"{code}.TW", f"{code}.TWO"]

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    errors = []

    for ticker in tickers_to_try:
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
                timeout=20,
            )

            if df is None or df.empty:
                errors.append(f"{ticker} 無歷史資料")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.loc[~df.index.duplicated(keep="last")].copy()
            df = df.sort_index()

            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            required = ["High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required):
                errors.append(f"{ticker} 缺少必要欄位")
                continue

            df = df.dropna(subset=required)
            df = df[df["Close"] > 0]
            df = df[df["Volume"] >= 0]

            if df.empty:
                errors.append(f"{ticker} 清理後沒有有效資料")
                continue

            df.attrs["ticker"] = ticker
            df.attrs["source"] = "Yahoo Finance"
            return df

        except Exception as exc:
            errors.append(f"{ticker}：{exc}")

    raise ValueError("；".join(errors) or "無法取得股價資料")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo_fundamentals(stock_id):
    """【修正】捨棄 Yahoo 靜態高低價，採用歷史 K 線重算，並修正市值異常。"""
    code = normalize_stock_id(stock_id)
    errors = []

    for suffix in (".TW", ".TWO"):
        ticker = f"{code}{suffix}"
        try:
            obj = yf.Ticker(ticker)
            info = obj.info

            if not info or info.get("quoteType") == "NONE" or "symbol" not in info:
                errors.append(f"{ticker} 無財務資訊")
                continue

            name = info.get("longName") or info.get("shortName") or info.get("displayName") or ""
            
            eps = safe_float(info.get("trailingEps"))
            market_cap = safe_float(info.get("marketCap"))
            shares_out = safe_float(info.get("sharesOutstanding"))

            high = safe_float(info.get("fiftyTwoWeekHigh"))
            low = safe_float(info.get("fiftyTwoWeekLow"))
            latest_price = None

            # -------------------------------------------------------------
            # 【財工核心修正】：強制捨棄 Yahoo 的錯誤數據，完全使用真實歷史股價重算
            # -------------------------------------------------------------
            try:
                hist = download_history(
                    code,
                    (datetime.today() - timedelta(days=400)).date(),
                    datetime.today().date(),
                )
                if not hist.empty:
                    # 強制覆蓋高低價
                    high = safe_float(hist["High"].max())
                    low = safe_float(hist["Low"].min())
                    latest_price = safe_float(hist["Close"].iloc[-1])
                    
                    # 修正市值 (若有發行股數，用真實收盤價重算市值，避免 ADR 匯率換算錯誤)
                    if shares_out and latest_price:
                        market_cap = latest_price * shares_out
                        
            except Exception:
                pass

            if name or eps is not None or market_cap is not None:
                return {
                    "ticker": ticker,
                    "name": name,
                    "eps": eps,
                    "market_cap": market_cap,
                    "high_52w": high,
                    "low_52w": low,
                    "latest_price": latest_price, # 傳遞給前端反推 EPS 用
                    "industry": info.get("industry", ""),
                    "summary": info.get("longBusinessSummary", ""),
                    "currency": info.get("currency", "TWD"),
                    "source": "Yahoo Finance (已由演算法修正)",
                    "error": "",
                }

        except Exception as exc:
            errors.append(f"{ticker}：{exc}")

    return {
        "ticker": "", "name": "", "eps": None, "market_cap": None, 
        "high_52w": None, "low_52w": None, "latest_price": None,
        "industry": "", "summary": "", "currency": "TWD", 
        "source": "Yahoo Finance", "error": "；".join(errors) or "無法取得財務資訊",
    }


def translate_to_zh(text, company_name="該公司"):
    """使用 Ollama 翻譯，並動態傳入公司名稱防止 AI 幻覺捏造企業名"""
    if not text:
        return "暫無公司簡介。"
    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是繁體中文財經翻譯員。"
                        "請忠實翻譯原文，不要新增或猜測公司財務數據。"
                        f"【重要指令】：請將原文中的主體公司名稱，一律翻譯為「{company_name}」，絕對不可以捏造或代入其他企業（如華為、Innoviz等）。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"請翻譯成繁體中文：\n{text}",
                },
            ],
            options={"temperature": 0},
        )
        return response["message"]["content"].strip() or text
    except Exception:
        return text


def format_market_cap(value):
    if value is None or value <= 0:
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.2f} 兆元"
    if value >= 1e8:
        return f"{value / 1e8:.2f} 億元"
    return f"{value:,.0f} 元"


# =========================================================
# 5. 技術指標計算與背離偵測
# =========================================================
def calculate_indicators(df):
    df = df.copy()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    close = pd.to_numeric(df[price_col], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    df["Price"] = close
    df["MA5"] = close.rolling(5, min_periods=5).mean()
    df["MA20"] = close.rolling(20, min_periods=20).mean()
    df["MA60"] = close.rolling(60, min_periods=60).mean()

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()

    df["MACD_DIF"] = ema12 - ema26
    df["MACD_DEM"] = df["MACD_DIF"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["MACD_OSC"] = df["MACD_DIF"] - df["MACD_DEM"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI"] = 50

    df["Vol_MA5"] = volume.rolling(5, min_periods=5).mean()
    df["Vol_MA20"] = volume.rolling(20, min_periods=20).mean()
    df["Vol_Ratio"] = volume / df["Vol_MA5"].replace(0, np.nan)

    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * volume).cumsum()
    df["OBV_MA10"] = df["OBV"].rolling(10, min_periods=10).mean()

    return df


def detect_rsi_divergence(df):
    """計算 RSI 高檔與低檔背離點位"""
    df = df.dropna(subset=["Price", "RSI"]).copy()
    df["Return_5d"] = df["Price"].shift(-5) / df["Price"] - 1
    df["Return_10d"] = df["Price"].shift(-10) / df["Price"] - 1

    window = 5
    peaks, troughs = [], []
    for i in range(window, len(df) - window):
        current_price = df["Price"].iloc[i]
        window_slice = df["Price"].iloc[i-window:i+window+1]
        if current_price == window_slice.max() and df["RSI"].iloc[i] > 60:
            peaks.append((i, df.index[i], current_price, df["RSI"].iloc[i]))
        if current_price == window_slice.min() and df["RSI"].iloc[i] < 40:
            troughs.append((i, df.index[i], current_price, df["RSI"].iloc[i]))

    bearish_divs, bullish_divs = [], []
    for j in range(1, len(peaks)):
        prev_idx, prev_date, prev_price, prev_rsi = peaks[j-1]
        curr_idx, curr_date, curr_price, curr_rsi = peaks[j]
        if curr_price > prev_price and curr_rsi < prev_rsi:
            bearish_divs.append({
                "Date": curr_date, "Price": curr_price, "RSI": curr_rsi,
                "5d_Return": df["Return_5d"].iloc[curr_idx],
                "10d_Return": df["Return_10d"].iloc[curr_idx]
            })

    for j in range(1, len(troughs)):
        prev_idx, prev_date, prev_price, prev_rsi = troughs[j-1]
        curr_idx, curr_date, curr_price, curr_rsi = troughs[j]
        if curr_price < prev_price and curr_rsi > prev_rsi:
            bullish_divs.append({
                "Date": curr_date, "Price": curr_price, "RSI": curr_rsi,
                "5d_Return": df["Return_5d"].iloc[curr_idx],
                "10d_Return": df["Return_10d"].iloc[curr_idx]
            })

    return pd.DataFrame(bearish_divs), pd.DataFrame(bullish_divs)


# =========================================================
# 6. PostgreSQL 歷史診斷
# =========================================================
def load_data_from_db(stock_id):
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="stock_db",
            user="postgres",
            password="postgres",
            connect_timeout=5,
        )
        query = """
            SELECT
                trade_date,
                analysis_time AS "診斷時間",
                close_price AS "收盤價",
                ma_short AS "MA5",
                ma_long AS "MA20",
                rsi14 AS "RSI",
                macd_osc AS "MACD_OSC",
                ai_advice AS "AI_綜合短評"
            FROM stock_strategy_analysis
            WHERE stock_id = %s
            ORDER BY analysis_time DESC
            LIMIT 100
        """
        return pd.read_sql_query(query, conn, params=(stock_id,))
    except Exception:
        return pd.DataFrame()
    finally:
        if conn is not None:
            conn.close()


def save_diagnosis_to_db(ticker, date_val, close_p, ma5, ma20, rsi, macd_dif, macd_dem, macd_osc, ai_advice):
    """將最新的診斷結果寫入資料庫"""
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, database="stock_db", user="postgres", password="postgres", connect_timeout=5
        )
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO stock_strategy_analysis 
            (stock_id, trade_date, analysis_time, close_price, ma_short, ma_long, rsi14, macd_dif, macd_dem, macd_osc, per, dividend_yield, pbr, ai_advice)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (ticker, date_val, datetime.now(), close_p, ma5, ma20, rsi, macd_dif, macd_dem, macd_osc, '0', '0', '0', ai_advice)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# =========================================================
# 7. 側邊欄
# =========================================================
st.sidebar.header("⚙️ 導航與選單")

if st.sidebar.button("🧹 清除系統快取 (重置數據)"):
    st.cache_data.clear()
    if "fundamental_result" in st.session_state:
        del st.session_state.fundamental_result
    st.sidebar.success("快取已清空！")

main_category = st.sidebar.selectbox(
    "🎯 選擇主分析維度",
    ["🏢 基本面分析", "📊 籌碼面分析", "📈 技術面分析"],
)

if main_category == "🏢 基本面分析":
    sub_feature = "公司基本面與財務資料"
elif main_category == "📊 籌碼面分析":
    sub_feature = st.sidebar.radio(
        "📌 子功能選擇",
        [
            "🏛️ 券商分點買賣超 (Yahoo直達)",
            "🔥 短線量價動能與資金流向",
        ],
    )
else:
    sub_feature = "個股與大盤技術診斷"

st.sidebar.markdown("---")
st.sidebar.subheader("📌 標的與時間設定")
stock_id_input = st.sidebar.text_input("請輸入台股代碼", value=st.session_state.stock_id).strip()
st.session_state.stock_id = stock_id_input

needs_date_picker = sub_feature in ["個股與大盤技術診斷", "🔥 短線量價動能與資金流向"]
if needs_date_picker:
    start_date = st.sidebar.date_input("開始日期", value=datetime.today().date() - timedelta(days=730))
    end_date = st.sidebar.date_input("結束日期", value=datetime.today().date())

# =========================================================
# 8. 基本面分析
# =========================================================
if main_category == "🏢 基本面分析":
    code = normalize_stock_id(stock_id_input)
    st.header(f"🏢 股票代碼 {code} 公司基本面與財務資料")

    if st.button("🔍 查詢精準基本面資料", use_container_width=True):
        with st.spinner("正在取得官方估值與財務資料..."):
            official = fetch_official_metrics(code)
            fundamental = fetch_yahoo_fundamentals(code)

            company_name = official.get("name") or fundamental.get("name") or f"台股 {code}"
            
            # 【修正】將正確的公司名稱傳入翻譯函式，壓制 AI 幻覺
            summary = translate_to_zh(fundamental.get("summary", ""), company_name)

            st.session_state.fundamental_result = {
                "code": code,
                "official": official,
                "fundamental": fundamental,
                "company_name": company_name,
                "summary": summary,
            }

            st.session_state.system_context = f"""
你是一位專業的台股基本面分析師。 目前標的：{code} {company_name}
官方估值來源：{official.get('source')} 本益比：{official.get('pe')} 股價淨值比：{official.get('pb')} 現金股利殖利率（%）：{official.get('yield')}
財務資料來源：{fundamental.get('source')} EPS：{fundamental.get('eps')} 市值：{fundamental.get('market_cap')} 52週最高價：{fundamental.get('high_52w')} 52週最低價：{fundamental.get('low_52w')} 公司簡介：{summary[:1500]}
只可依據已取得的數據回答，不得捏造即時行情、財報或公司資訊。 必須區分官方資料與第三方資料，資料不足時明確說明。 股票分析不可保證獲利。 
"""

    if "fundamental_result" in st.session_state:
        result = st.session_state.fundamental_result
        if result["code"] == code:
            official = result["official"]
            fundamental = result["fundamental"]

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown("**🏢 公司名稱**")
            c1.subheader(f"{code} {result['company_name']}")

            c2.markdown("**📡 官方資料來源**")
            c2.write(official.get("source", "N/A"))

            c3.metric("本益比 (P/E)", format_number(official.get("pe")))
            c4.metric("股價淨值比 (P/B)", format_number(official.get("pb")))

            st.caption(
                f"官方資料查詢時間：{official.get('raw_date', 'N/A')}。"
                "估值資料依官方 API 當時回傳內容呈現，不代表即時成交行情。"
            )

            if official.get("error"):
                st.warning("官方資料未能完整取得：" + official["error"])

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💰 官方評價與財務指標")
                dividend_yield = official.get("yield")
                st.write(
                    "**現金股利殖利率：** " + (f"{format_number(dividend_yield)}%" if dividend_yield is not None else "N/A")
                )
                
                # 【財工動態修正】：利用官方 PE 與真實收盤價反推真實 EPS
                official_pe = official.get("pe")
                latest_price = fundamental.get("latest_price")
                real_eps = fundamental.get("eps")
                
                if official_pe and latest_price and official_pe > 0:
                    real_eps = latest_price / official_pe
                    eps_label = "每股盈餘 (EPS) [官方本益比反推]"
                else:
                    eps_label = "每股盈餘 (EPS) [Yahoo]"

                st.write(f"**{eps_label}：** {format_number(real_eps)} 元")
                st.write(f"**公司總市值：** {format_market_cap(fundamental.get('market_cap'))}")
                st.caption("EPS 與市值已透過最新收盤價與官方數據進行動態校正。")

            with col2:
                st.subheader("📈 近 52 週價格區間")
                st.write(f"**52 週最高價：** {format_number(fundamental.get('high_52w'))} 元")
                st.write(f"**52 週最低價：** {format_number(fundamental.get('low_52w'))} 元")

            st.markdown("---")
            st.subheader("📄 公司簡介（繁體中文）")
            st.write(result["summary"])

            if fundamental.get("error"):
                st.warning("第三方財務資料未能完整取得：" + fundamental["error"])


# =========================================================
# 9. 籌碼面分析
# =========================================================
elif main_category == "📊 籌碼面分析":
    code = normalize_stock_id(stock_id_input)
    st.header(f"📊 股票代碼 {code} {sub_feature}")

    if sub_feature == "🏛️ 券商分點買賣超 (Yahoo直達)":
        st.subheader(f"🏛️ {code} 主力券商分點進出明細")
        st.info("券商分點資料請直接查看 Yahoo 股市原始頁面；本功能不會將未取得的資料偽裝成已完成的籌碼分析。")
        yahoo_broker_url = f"https://tw.stock.yahoo.com/quote/{code}.TW/broker-trading"
        st.link_button(f"🚀 開啟 Yahoo 股市 {code} 券商分點明細", yahoo_broker_url, use_container_width=True)

        st.session_state.system_context = f"""
你是一位台股籌碼面分析師。 使用者正在查詢 {code} 的券商分點資料。 若沒有實際取得分點買賣超數據，不得推測主力買賣方向。 
"""

    elif sub_feature == "🔥 短線量價動能與資金流向":
        if st.sidebar.button("🚀 執行短線籌碼量價分析", use_container_width=True):
            if start_date >= end_date:
                st.error("結束日期必須大於開始日期！")
            else:
                try:
                    with st.spinner("正在下載股價與成交量..."):
                        raw = download_history(code, start_date, end_date)
                        df_chip = calculate_indicators(raw)
                        st.session_state.df_chip = df_chip
                        st.session_state.chip_ticker = raw.attrs.get("ticker", code)
                except Exception as exc:
                    st.error(f"籌碼分析失敗：{exc}")

        if "df_chip" in st.session_state:
            df_chip = st.session_state.df_chip
            price_col = "Price"

            if len(df_chip) < 2:
                st.warning("歷史資料不足，無法計算價格變化。")
            else:
                latest = df_chip.iloc[-1]
                previous = df_chip.iloc[-2]

                latest_price = safe_float(latest[price_col])
                previous_price = safe_float(previous[price_col])
                latest_volume = safe_float(latest["Volume"])
                volume_ratio = safe_float(latest["Vol_Ratio"])

                price_change = (
                    (latest_price / previous_price - 1) * 100
                    if latest_price is not None and previous_price is not None and previous_price != 0
                    else None
                )

                volume_text = f"{latest_volume / 1000:,.0f} 張" if latest_volume is not None else "N/A"
                ratio_text = f"量比 {volume_ratio:.2f}" if volume_ratio is not None else "量比 N/A"

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "📌 最新收盤價",
                    f"{format_number(latest_price)} 元",
                    f"{price_change:+.2f}%" if price_change is not None else None,
                )

                c2.metric("📦 最新成交量", volume_text, ratio_text)

                c3.metric(
                    "🔥 量能動態",
                    (
                        "爆量" if volume_ratio is not None and volume_ratio > 1.5
                        else "縮量" if volume_ratio is not None and volume_ratio < 0.8
                        else "量能正常" if volume_ratio is not None else "資料不足"
                    ),
                )

                obv = safe_float(latest["OBV"])
                obv_ma = safe_float(latest["OBV_MA10"])

                obv_status = (
                    "🟢 OBV 高於均線" if obv is not None and obv_ma is not None and obv > obv_ma
                    else "🔴 OBV 低於均線" if obv is not None and obv_ma is not None
                    else "資料不足"
                )

                c4.metric("🌊 OBV 趨勢", obv_status)

                st.markdown("---")

                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                    subplot_titles=("股價與成交量", "OBV 能量潮與 10 日均線"),
                )

                fig.add_trace(
                    go.Scatter(x=df_chip.index, y=df_chip["Price"], mode="lines", name="收盤價", line=dict(color="royalblue", width=2)),
                    row=1, col=1,
                )

                colors = [
                    "red" if i == 0 or df_chip["Price"].iloc[i] >= df_chip["Price"].iloc[i - 1] else "green"
                    for i in range(len(df_chip))
                ]

                fig.add_trace(
                    go.Bar(x=df_chip.index, y=df_chip["Volume"], name="成交量（股）", marker_color=colors, opacity=0.35),
                    row=1, col=1,
                )

                fig.add_trace(
                    go.Scatter(x=df_chip.index, y=df_chip["Vol_MA5"], name="5 日均量", line=dict(color="orange")),
                    row=1, col=1,
                )

                fig.add_trace(
                    go.Scatter(x=df_chip.index, y=df_chip["OBV"], name="OBV", line=dict(color="purple")),
                    row=2, col=1,
                )

                fig.add_trace(
                    go.Scatter(x=df_chip.index, y=df_chip["OBV_MA10"], name="OBV 10 日均線", line=dict(color="gray", dash="dash")),
                    row=2, col=1,
                )

                fig.update_layout(height=700, hovermode="x unified", template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

                st.session_state.system_context = f"""
你是一位專業的台股量價分析師。 股票：{code} 資料來源：Yahoo Finance 最新收盤價：{format_number(latest_price)} 價格變化：{format_number(price_change)}% 成交量：{volume_text} 量比：{format_number(volume_ratio)} OBV 狀態：{obv_status}
請依據這些實際數據解釋量價變化。 不得將 OBV 直接說成法人買賣超或主力實際持倉。 
"""

# =========================================================
# 10. 技術面分析
# =========================================================
elif main_category == "📈 技術面分析":
    code = normalize_stock_id(stock_id_input)
    st.header(f"📈 股票代碼 {code} 個股與大盤技術診斷")

    if st.sidebar.button("🚀 執行量化指標與 AI 診斷", use_container_width=True):
        if start_date >= end_date:
            st.error("結束日期必須大於開始日期！")
        else:
            try:
                with st.spinner("正在取得個股、大盤、匯率並呼叫 Ollama 生成 AI 診斷..."):
                    raw = download_history(code, start_date, end_date)
                    df = calculate_indicators(raw)
                    market = download_history("^TWII", start_date, end_date)
                    fx = download_history("TWD=X", start_date, end_date)

                    bearish_df, bullish_df = detect_rsi_divergence(df)
                    st.session_state.bearish_df = bearish_df
                    st.session_state.bullish_df = bullish_df

                    price_col = "Adj Close" if "Adj Close" in market.columns else "Close"
                    market_close = market[price_col].dropna()

                    if not market_close.empty:
                        market_latest = float(market_close.iloc[-1])
                        market_prev = float(market_close.iloc[-2]) if len(market_close) > 1 else market_latest
                        market_change = ((market_latest / market_prev - 1) * 100) if market_prev else 0
                        market_ma20 = float(market_close.rolling(20).mean().iloc[-1]) if len(market_close) >= 20 else None
                        market_trend = "站上月線 (偏多)" if market_ma20 and market_latest >= market_ma20 else "跌破月線 (偏空)"
                    else:
                        market_latest, market_change, market_trend = None, None, "資料不足"

                    fx_close = fx["Close"].dropna() if not fx.empty and "Close" in fx.columns else pd.Series(dtype=float)
                    if not fx_close.empty:
                        fx_latest = float(fx_close.iloc[-1])
                        fx_prev = float(fx_close.iloc[-6]) if len(fx_close) >= 6 else float(fx_close.iloc[0])
                        fx_change = ((fx_latest / fx_prev - 1) * 100) if fx_prev else None
                    else:
                        fx_latest, fx_change = None, None

                    st.session_state.macro_info = {
                        "market_latest": market_latest,
                        "market_change": market_change,
                        "market_trend": market_trend,
                        "fx_latest": fx_latest,
                        "fx_change": fx_change,
                    }

                    latest = df.iloc[-1]
                    close_price = round(float(latest["Price"]), 2)
                    
                    ai_prompt = f"""
                    妳是一位專業的台股量化投資分析師。請嚴格使用繁體中文，總字數 120 字內。
                    【即時數據】股票:{code} | 收盤:{close_price} | MA5:{latest['MA5']:.2f} | MA20:{latest['MA20']:.2f} | RSI:{latest['RSI']:.2f} | MACD_OSC:{latest['MACD_OSC']:.2f}
                    【市場環境】大盤加權指數:{market_latest:.0f} ({market_trend}) | 美元/台幣匯率:{fx_latest:.2f}
                    請綜合個股技術面與總體市場環境：
                    1. 📢 綜合診斷：
                    2. 💡 給老闆的悄悄話：
                    """
                    
                    ai_advice = "AI 未能成功產生建議。"
                    try:
                        resp = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": "qwen2.5:3b", "prompt": ai_prompt, "stream": False},
                            timeout=30
                        )
                        if resp.status_code == 200:
                            ai_advice = resp.json().get("response", "").strip()
                    except Exception:
                        ai_advice = "本機 Ollama 未回應。"

                    st.session_state.latest_ai_advice = ai_advice
                    save_diagnosis_to_db(code, latest.name.date(), close_price, float(latest['MA5']), float(latest['MA20']), float(latest['RSI']), float(latest['MACD_DIF']), float(latest['MACD_DEM']), float(latest['MACD_OSC']), ai_advice)

                    st.session_state.df = df
                    st.session_state.ticker = raw.attrs.get("ticker", code)

                    if hasattr(load_data_from_db, "clear"):
                        load_data_from_db.clear()

            except Exception as exc:
                st.error(f"技術分析執行失敗：{exc}")

    if "df" in st.session_state:
        df = st.session_state.df
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest

        latest_price = safe_float(latest["Price"])
        prev_price = safe_float(previous["Price"])
        change = ((latest_price / prev_price - 1) * 100 if latest_price is not None and prev_price is not None and prev_price != 0 else None)

        macro = st.session_state.get("macro_info", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📌 個股最新收盤", f"{format_number(latest_price)} 元", f"{change:+.2f}%" if change is not None else None)
        c2.metric("🏛️ 加權指數", format_number(macro.get("market_latest"), 0), f"{macro['market_change']:+.2f}%" if macro.get("market_change") is not None else None)
        c3.metric("💵 美元/台幣匯率", format_number(macro.get("fx_latest")), f"區間變化 {macro['fx_change']:+.2f}%" if macro.get("fx_change") is not None else None)
        c4.metric("📈 大盤月線趨勢", macro.get("market_trend", "資料不足"))

        st.markdown("---")

        db_df = load_data_from_db(code)
        ai_dict = {}
        if not db_df.empty:
            for _, row in db_df.iterrows():
                d_str = str(row['trade_date'])
                if d_str not in ai_dict:
                    ai_dict[d_str] = row['AI_綜合短評']

        hover_texts = []
        for date, row in df.iterrows():
            d_str = date.strftime('%Y-%m-%d')
            price = row["Price"]
            ma5 = row['MA5']
            rsi = row['RSI']
            macd_osc = row['MACD_OSC']
            
            ai_text = ai_dict.get(d_str, "")
            if d_str == df.index[-1].strftime('%Y-%m-%d') and "latest_ai_advice" in st.session_state:
                ai_text = st.session_state.latest_ai_advice

            ai_html = ""
            if ai_text and ai_text != "None":
                raw_ai_text = str(ai_text).replace("\n", " ").replace("2. 💡", "<br><br>2. 💡")
                parts = raw_ai_text.split("<br>")
                wrapped_parts = []
                for part in parts:
                    if part.strip():
                        wrapped_part = "<br>".join([part[i:i+28] for i in range(0, len(part), 28)])
                        wrapped_parts.append(wrapped_part)
                    else:
                        wrapped_parts.append("")
                ai_html = f"<br>--------------------------------------------------<br><b>🤖 AI 綜合診斷與短評:</b><br>" + "<br>".join(wrapped_parts)

            trend = "🟢 偏多 (+)" if (price >= ma5 and macd_osc >= 0) else "🔴 偏空 (-)"
            hover_msg = (
                f"<b>🕒 時間</b>: {d_str}<br><b>💰 收盤價</b>: {price:.2f} 元<br>"
                f"<b>📊 多空訊號</b>: {trend}<br><b>📈 RSI(14)</b>: {rsi:.2f}<br>"
                f"<b>📉 MACD OSC</b>: {macd_osc:.4f}{ai_html}"
            )
            hover_texts.append(hover_msg)
            
        df['hover_text'] = hover_texts

        bearish_df = st.session_state.get("bearish_df", pd.DataFrame())
        bullish_df = st.session_state.get("bullish_df", pd.DataFrame())

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.20],
            subplot_titles=(f"{st.session_state.ticker} 歷史走勢與技術分析", "MACD 指標", "RSI 指標 (包含背離訊號)"),
        )

        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Price"], name="收盤價", line=dict(color="royalblue", width=2),
                hovertemplate="%{text}<extra></extra>", text=df['hover_text'],
                hoverlabel=dict(bgcolor="white", font=dict(size=12, color="black"))
            ),
            row=1, col=1,
        )

        for col, color in [("MA5", "orange"), ("MA20", "green"), ("MA60", "purple")]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=1), hoverinfo="skip"), row=1, col=1)

        if not bearish_df.empty:
            fig.add_trace(go.Scatter(x=bearish_df['Date'], y=bearish_df['Price'], mode='markers', name='高檔背離', marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)
        if not bullish_df.empty:
            fig.add_trace(go.Scatter(x=bullish_df['Date'], y=bullish_df['Price'], mode='markers', name='低檔背離', marker=dict(symbol='triangle-up', size=12, color='green')), row=1, col=1)

        macd_colors = ["red" if value >= 0 else "green" for value in df["MACD_OSC"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_OSC"], name="MACD 柱狀圖", marker_color=macd_colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_DIF"], name="DIF", line=dict(color="blue")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_DEM"], name="DEA", line=dict(color="orange")), row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        if not bearish_df.empty:
            fig.add_trace(go.Scatter(x=bearish_df['Date'], y=bearish_df['RSI'], mode='markers', showlegend=False, marker=dict(symbol='triangle-down', size=8, color='red')), row=3, col=1)
        if not bullish_df.empty:
            fig.add_trace(go.Scatter(x=bullish_df['Date'], y=bullish_df['RSI'], mode='markers', showlegend=False, marker=dict(symbol='triangle-up', size=8, color='green')), row=3, col=1)

        fig.update_layout(height=850, template="plotly_white", hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 查看詳細背離歷史紀錄 (RSI)"):
            col1, col2 = st.columns(2)
            with col1:
                st.error("🔴 高檔背離紀錄")
                st.dataframe(bearish_df, use_container_width=True)
            with col2:
                st.success("🟢 低檔背離紀錄")
                st.dataframe(bullish_df, use_container_width=True)

        st.subheader("📋 最新技術指標")
        indicators = pd.DataFrame({
            "指標": ["MA5", "MA20", "MA60", "MACD_DIF", "MACD_DEM", "MACD_OSC", "RSI", "成交量", "5日均量", "量比", "OBV"],
            "最新數值": [
                latest.get("MA5"), latest.get("MA20"), latest.get("MA60"), latest.get("MACD_DIF"),
                latest.get("MACD_DEM"), latest.get("MACD_OSC"), latest.get("RSI"), latest.get("Volume"),
                latest.get("Vol_MA5"), latest.get("Vol_Ratio"), latest.get("OBV")
            ]
        })
        st.dataframe(indicators, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader(f"🗄️ PostgreSQL 歷史診斷紀錄 (股票代碼: {code})")
        if not db_df.empty:
            st.dataframe(db_df, use_container_width=True, hide_index=True)
        else:
            st.info("目前尚無此股票的資料庫歷程紀錄，或 PostgreSQL 未連線。")

        st.session_state.system_context = f"""
你是一位專業的台股量化技術分析師。 標的：{code} 資料來源：Yahoo Finance 最後交易日期：{df.index[-1].date()} 最新收盤價：{format_number(latest_price)} 股價漲跌幅：{format_number(change)}% MA5：{format_number(safe_float(latest['MA5']))} MA20：{format_number(safe_float(latest['MA20']))} MA60：{format_number(safe_float(latest['MA60']))} MACD DIF：{format_number(safe_float(latest['MACD_DIF']))} MACD DEA：{format_number(safe_float(latest['MACD_DEM']))} MACD OSC：{format_number(safe_float(latest['MACD_OSC']))} RSI：{format_number(safe_float(latest['RSI']))} 大盤趨勢：{macro.get('market_trend', '資料不足')} 美元台幣匯率：{format_number(macro.get('fx_latest'))}
請根據實際指標分析趨勢、風險及可能情境。 不得聲稱能準確預測未來價格，也不得捏造法人籌碼資料。 
"""
    else:
        st.info("👈 請在左側點擊「🚀 執行量化指標與 AI 診斷」以載入數據。")

# =========================================================
# 11. 全域 AI 對話
# =========================================================
st.markdown("---")
st.subheader("🤖 全域 AI 投資顧問")

# 切換選單改為 3.5 Flash，並直接從 secrets 讀取金鑰
ai_model_option = st.radio(
    "⚙️ 切換語言模型：",
    ["Qwen 2.5 (本機 Ollama)", "Gemini 3.5 Flash (雲端 API)"],
    horizontal=True
)

gemini_api_key = st.secrets["GEMINI_API_KEY"]

for msg in st.session_state.messages:
    if msg.get("role") in ("user", "assistant"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

prompt = st.chat_input("向 AI 顧問提問...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"{ai_model_option} 正在為您分析..."):
            try:
                reply = ""
                
                if ai_model_option == "Qwen 2.5 (本機 Ollama)":
                    messages_payload = [{"role": "system", "content": st.session_state.system_context}]
                    messages_payload.extend(st.session_state.messages[-12:])

                    response = ollama.chat(
                        model="qwen2.5:3b",
                        messages=messages_payload,
                        options={"temperature": 0.2},
                    )
                    reply = response["message"]["content"].strip()

                elif ai_model_option == "Gemini 3.5 Flash (雲端 API)":
                    # 直接使用金鑰，不需要再判斷輸入框
                    genai.configure(api_key=gemini_api_key)
                    
                    # 模型參數字串更新為 'gemini-3.5-flash'
                    model = genai.GenerativeModel(
                        model_name='gemini-3.5-flash',
                        system_instruction=st.session_state.system_context
                    )
                    
                    gemini_history = []
                    for m in st.session_state.messages[-12:]:
                        role = "user" if m["role"] == "user" else "model"
                        gemini_history.append({"role": role, "parts": [m["content"]]})
                        
                    response = model.generate_content(
                        gemini_history,
                        generation_config=genai.GenerationConfig(temperature=0.2)
                    )
                    reply = response.text.strip()

                if not reply:
                    reply = "AI 未回傳內容，請稍後再試。"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as exc:
                if ai_model_option == "Qwen 2.5 (本機 Ollama)":
                    error_message = f"Ollama 連線失敗：{exc}\n\n請確認 Ollama 服務正在執行，並已下載 qwen2.5:3b 模型。"
                else:
                    error_message = f"Gemini API 呼叫失敗：{exc}\n\n請確認您的 API Key 是否正確以及網路狀態。"
                st.error(error_message)
