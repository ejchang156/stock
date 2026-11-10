from datetime import datetime, timedelta
import logging
import time

import google.generativeai as genai
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
# 1. 頁面設定與初始化 (適應手機螢幕)
# =========================================================
st.set_page_config(
    page_title="全能 AI 量化投資助理",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed", # 手機端預設收合側邊欄，提供最大閱讀視野
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
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 15

st.title("📈 全能 AI 量化投資助理")
st.caption("官方基本面資料、歷史股價、技術指標、量價分析與 AI 智慧診斷")
st.markdown("---")

# 讀取 Gemini API Key Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")


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
    """採用歷史 K 線重算 52 週高低價與市值修正。"""
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

            try:
                hist = download_history(
                    code,
                    (datetime.today() - timedelta(days=400)).date(),
                    datetime.today().date(),
                )
                if not hist.empty:
                    high = safe_float(hist["High"].max())
                    low = safe_float(hist["Low"].min())
                    latest_price = safe_float(hist["Close"].iloc[-1])
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
                    "latest_price": latest_price,
                    "industry": info.get("industry", ""),
                    "summary": info.get("longBusinessSummary", ""),
                    "currency": info.get("currency", "TWD"),
                    "source": "Yahoo Finance (已校正)",
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
    """使用 Gemini API 翻譯公司簡介為繁體中文。"""
    if not text:
        return "暫無公司簡介。"
    if not GEMINI_API_KEY:
        return text

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=f"你是繁體中文財經翻譯員。請忠實翻譯原文，將主體公司名稱統一翻譯為「{company_name}」，切勿新增未提到的財務數據。"
        )
        response = model.generate_content(f"請翻譯成繁體中文：\n{text}")
        return response.text.strip()
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
# 6. PostgreSQL 歷史診斷 (相容雲端 DB / 自動防呆)
# =========================================================
def get_db_connection():
    """優先嘗試從 st.secrets 讀取雲端資料庫設定。"""
    try:
        db = st.secrets.get("postgres")
        if db:
            return psycopg2.connect(
                host=db["host"],
                port=db.get("port", 5432),
                database=db["database"],
                user=db["user"],
                password=db["password"],
                sslmode="require",
                connect_timeout=5,
            )
        # 本機退路
        return psycopg2.connect(
            host="localhost", port=5432, database="stock_db", user="postgres", password="postgres", connect_timeout=3
        )
    except Exception:
        return None


def load_data_from_db(stock_id):
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
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
        df = pd.read_sql_query(query, conn, params=(stock_id,))
        conn.close()
        return df
    except Exception:
        if conn:
            conn.close()
        return pd.DataFrame()


def save_diagnosis_to_db(ticker, date_val, close_p, ma5, ma20, rsi, macd_dif, macd_dem, macd_osc, ai_advice):
    conn = get_db_connection()
    if conn is None:
        return
    try:
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
        if conn:
            conn.close()


# =========================================================
# 7. 側邊欄控制
# =========================================================
st.sidebar.header("⚙️ 導航與選單")

if st.sidebar.button("🧹 清除系統快取", use_container_width=True):
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
# 8. 基本面分析 (手機自適應排版)
# =========================================================
if main_category == "🏢 基本面分析":
    code = normalize_stock_id(stock_id_input)
    st.header(f"🏢 {code} 公司基本面")

    if st.button("🔍 查詢精準基本面資料", use_container_width=True):
        with st.spinner("正在取得官方估值與財務資料..."):
            official = fetch_official_metrics(code)
            fundamental = fetch_yahoo_fundamentals(code)
            company_name = official.get("name") or fundamental.get("name") or f"台股 {code}"
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
只可依據已取得的數據回答，不得捏造即時行情。 
"""

    if "fundamental_result" in st.session_state:
        result = st.session_state.fundamental_result
        if result["code"] == code:
            official = result["official"]
            fundamental = result["fundamental"]

            # 手機端採 2 欄排列
            c1, c2 = st.columns(2)
            c1.markdown("**🏢 公司名稱**")
            c1.subheader(f"{code} {result['company_name']}")
            c2.markdown("**📡 官方來源**")
            c2.write(official.get("source", "N/A"))

            c3, c4 = st.columns(2)
            c3.metric("本益比 (P/E)", format_number(official.get("pe")))
            c4.metric("股價淨值比 (P/B)", format_number(official.get("pb")))

            st.caption(f"官方資料查詢時間：{official.get('raw_date', 'N/A')}。")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💰 官方與財務指標")
                dividend_yield = official.get("yield")
                st.write("**現金股利殖利率：** " + (f"{format_number(dividend_yield)}%" if dividend_yield is not None else "N/A"))
                
                official_pe = official.get("pe")
                latest_price = fundamental.get("latest_price")
                real_eps = fundamental.get("eps")
                
                if official_pe and latest_price and official_pe > 0:
                    real_eps = latest_price / official_pe
                    eps_label = "EPS (官方 PE 反推)"
                else:
                    eps_label = "EPS (Yahoo)"

                st.write(f"**{eps_label}：** {format_number(real_eps)} 元")
                st.write(f"**公司總市值：** {format_market_cap(fundamental.get('market_cap'))}")

            with col2:
                st.subheader("📈 近 52 週價格區間")
                st.write(f"**52 週最高價：** {format_number(fundamental.get('high_52w'))} 元")
                st.write(f"**52 週最低價：** {format_number(fundamental.get('low_52w'))} 元")

            st.markdown("---")
            st.subheader("📄 公司簡介")
            st.write(result["summary"])


# =========================================================
# 9. 籌碼面分析
# =========================================================
elif main_category == "📊 籌碼面分析":
    code = normalize_stock_id(stock_id_input)
    st.header(f"📊 {code} {sub_feature}")

    if sub_feature == "🏛️ 券商分點買賣超 (Yahoo直達)":
        st.subheader(f"🏛️ {code} 主力券商分點進出明細")
        st.info("券商分點明細請點擊下方按鈕切換至專屬網頁查看。")
        yahoo_broker_url = f"https://tw.stock.yahoo.com/quote/{code}.TW/broker-trading"
        st.link_button(f"🚀 開啟 Yahoo 股市 {code} 分點明細", yahoo_broker_url, use_container_width=True)

    elif sub_feature == "🔥 短線量價動能與資金流向":
        if st.button("🚀 執行短線籌碼量價分析", use_container_width=True):
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

            if len(df_chip) >= 2:
                latest = df_chip.iloc[-1]
                previous = df_chip.iloc[-2]

                latest_price = safe_float(latest[price_col])
                previous_price = safe_float(previous[price_col])
                latest_volume = safe_float(latest["Volume"])
                volume_ratio = safe_float(latest["Vol_Ratio"])

                price_change = ((latest_price / previous_price - 1) * 100) if latest_price and previous_price else None
                volume_text = f"{latest_volume / 1000:,.0f} 張" if latest_volume is not None else "N/A"
                ratio_text = f"量比 {volume_ratio:.2f}" if volume_ratio is not None else "量比 N/A"

                c1, c2 = st.columns(2)
                c1.metric("📌 最新收盤價", f"{format_number(latest_price)} 元", f"{price_change:+.2f}%" if price_change else None)
                c2.metric("📦 最新成交量", volume_text, ratio_text)

                c3, c4 = st.columns(2)
                c3.metric(
                    "🔥 量能動態",
                    "爆量" if volume_ratio and volume_ratio > 1.5 else "縮量" if volume_ratio and volume_ratio < 0.8 else "正常"
                )
                obv = safe_float(latest["OBV"])
                obv_ma = safe_float(latest["OBV_MA10"])
                obv_status = "🟢 高於均線" if obv and obv_ma and obv > obv_ma else "🔴 低於均線" if obv and obv_ma else "資料不足"
                c4.metric("🌊 OBV 趨勢", obv_status)

                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                    subplot_titles=("股價與成交量", "OBV 能量潮"),
                )

                fig.add_trace(go.Scatter(x=df_chip.index, y=df_chip["Price"], mode="lines", name="收盤價", line=dict(color="royalblue")), row=1, col=1)
                colors = ["red" if i == 0 or df_chip["Price"].iloc[i] >= df_chip["Price"].iloc[i - 1] else "green" for i in range(len(df_chip))]
                fig.add_trace(go.Bar(x=df_chip.index, y=df_chip["Volume"], name="成交量", marker_color=colors, opacity=0.35), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chip.index, y=df_chip["OBV"], name="OBV", line=dict(color="purple")), row=2, col=1)

                fig.update_layout(height=500, hovermode="x unified", template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
                # 手機圖表停用 drag 手勢避免滑動鎖定
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


# =========================================================
# 10. 技術面分析
# =========================================================
elif main_category == "📈 技術面分析":
    code = normalize_stock_id(stock_id_input)
    st.header(f"📈 {code} 個股與大盤技術診斷")

    if st.button("🚀 執行量化指標與 AI 診斷", use_container_width=True):
        if start_date >= end_date:
            st.error("結束日期必須大於開始日期！")
        else:
            try:
                with st.spinner("正在計算技術指標並生成 AI 診斷..."):
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
                    fx_latest = float(fx_close.iloc[-1]) if not fx_close.empty else None

                    st.session_state.macro_info = {
                        "market_latest": market_latest,
                        "market_change": market_change,
                        "market_trend": market_trend,
                        "fx_latest": fx_latest,
                    }

                    latest = df.iloc[-1]
                    close_price = round(float(latest["Price"]), 2)
                    
                    ai_prompt = f"""
                    你是專業的台股量化投資分析師。請嚴格使用繁體中文，字數 120 字內。
                    【即時數據】股票:{code} | 收盤:{close_price} | MA5:{latest['MA5']:.2f} | MA20:{latest['MA20']:.2f} | RSI:{latest['RSI']:.2f} | MACD_OSC:{latest['MACD_OSC']:.2f}
                    【市場環境】大盤加權指數:{market_latest:.0f} ({market_trend}) | 美元/台幣匯率:{fx_latest:.2f}
                    請給予短評與建議：
                    1. 📢 綜合診斷：
                    2. 💡 給投資人的短評：
                    """
                    
                    ai_advice = "AI 未能成功產生建議。"
                    if GEMINI_API_KEY:
                        try:
                            genai.configure(api_key=GEMINI_API_KEY)
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            resp = model.generate_content(ai_prompt)
                            ai_advice = resp.text.strip()
                        except Exception as exc:
                            ai_advice = f"Gemini API 呼叫失敗：{exc}"

                    st.session_state.latest_ai_advice = ai_advice
                    save_diagnosis_to_db(code, latest.name.date(), close_price, float(latest['MA5']), float(latest['MA20']), float(latest['RSI']), float(latest['MACD_DIF']), float(latest['MACD_DEM']), float(latest['MACD_OSC']), ai_advice)

                    st.session_state.df = df
                    st.session_state.ticker = raw.attrs.get("ticker", code)

            except Exception as exc:
                st.error(f"技術分析執行失敗：{exc}")

    if "df" in st.session_state:
        df = st.session_state.df
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest

        latest_price = safe_float(latest["Price"])
        prev_price = safe_float(previous["Price"])
        change = ((latest_price / prev_price - 1) * 100) if latest_price and prev_price else None
        macro = st.session_state.get("macro_info", {})

        # 手機 2x2 矩陣指標排版
        c1, c2 = st.columns(2)
        c1.metric("📌 個股最新收盤", f"{format_number(latest_price)} 元", f"{change:+.2f}%" if change else None)
        c2.metric("🏛️ 加權指數", format_number(macro.get("market_latest"), 0), f"{macro['market_change']:+.2f}%" if macro.get("market_change") else None)

        c3, c4 = st.columns(2)
        c3.metric("💵 美元/台幣匯率", format_number(macro.get("fx_latest")))
        c4.metric("📈 大盤月線趨勢", macro.get("market_trend", "資料不足"))

        st.markdown("---")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.20],
            subplot_titles=(f"{st.session_state.ticker} 走勢", "MACD", "RSI"),
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["Price"], name="收盤價", line=dict(color="royalblue", width=2)), row=1, col=1)
        for col, color in [("MA5", "orange"), ("MA20", "green")]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=1)), row=1, col=1)

        macd_colors = ["red" if value >= 0 else "green" for value in df["MACD_OSC"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_OSC"], name="MACD 柱狀圖", marker_color=macd_colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")), row=3, col=1)

        fig.update_layout(height=600, template="plotly_white", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})

        st.subheader("📋 最新技術指標")
        indicators = pd.DataFrame({
            "指標": ["MA5", "MA20", "RSI", "MACD_OSC", "量比"],
            "數值": [latest.get("MA5"), latest.get("MA20"), latest.get("RSI"), latest.get("MACD_OSC"), latest.get("Vol_Ratio")]
        })
        st.dataframe(indicators, use_container_width=True, hide_index=True)


# =========================================================
# 11. 全域 AI 對話 (雲端 API / 本機彈性切換)
# =========================================================
st.markdown("---")
st.subheader("🤖 全域 AI 投資顧問")

ai_model_option = st.radio(
    "⚙️ 選擇 AI 引擎：",
    ["Gemini 2.5 Flash (雲端 API 推薦)", "Qwen 2.5 (本機 Ollama)"],
    horizontal=True
)

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
        with st.spinner(f"AI 正在思考中..."):
            try:
                reply = ""
                if ai_model_option.startswith("Gemini"):
                    if not GEMINI_API_KEY:
                        reply = "⚠️ 未設定 GEMINI_API_KEY，請於 Secrets 中配置 key。"
                    else:
                        genai.configure(api_key=GEMINI_API_KEY)
                        model = genai.GenerativeModel(
                            model_name='gemini-2.5-flash',
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

                else:
                    messages_payload = [{"role": "system", "content": st.session_state.system_context}]
                    messages_payload.extend(st.session_state.messages[-12:])
                    response = ollama.chat(
                        model="qwen2.5:3b",
                        messages=messages_payload,
                        options={"temperature": 0.2},
                    )
                    reply = response["message"]["content"].strip()

                if not reply:
                    reply = "AI 未回傳內容，請稍後再試。"

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as exc:
                st.error(f"AI 回應失敗：{exc}")