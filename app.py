# app.py
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  

# ---------------------------------------------------------------------
# Utilities & Numerics: Black-Scholes, Greeks, IV solver, GBM simulation
# ---------------------------------------------------------------------
def safe_scalar(x):
    """Convert pandas Series/np.array to scalar if necessary"""
    if isinstance(x, (pd.Series, pd.DataFrame, np.ndarray)):
        x = np.asarray(x).squeeze()
    return float(x)

def days_to_years(d):
    return d / 365.0

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """European Black-Scholes price. T in years, sigma annualized."""
    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    if sigma <= 0:
        if option_type == "call":
            return max(0.0, S - K * math.exp(-r * T))
        else:
            return max(0.0, K * math.exp(-r * T) - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def option_greeks(S, K, T, r, sigma, option_type="call"):
    """Return Delta, Gamma, Theta (per day), Vega, Rho"""
    if T <= 0 or sigma <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else (norm.cdf(d1) - 1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    if option_type == "put":
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r*T) * norm.cdf(-d2)) / 365.0
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0  # per 1% vol
    rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0
    return delta, gamma, theta, vega, rho

def implied_volatility(S, K, T, r, market_price, option_type="call"):
    """Find IV by root-finding. Returns NaN if failed."""
    if market_price <= 0 or T <= 0:
        return np.nan
    def obj(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    try:
        return brentq(obj, 1e-6, 5.0, maxiter=200)
    except Exception:
        try:
            return brentq(obj, 1e-8, 10.0, maxiter=200)
        except Exception:
            return np.nan

def gbm_simulate_paths(S0, mu, sigma, T_years, steps=252, n_paths=10000, seed=None):
    """Simulate GBM paths. Return numpy array (n_paths, steps+1)."""
    rng = np.random.default_rng(seed)
    dt = T_years / steps
    increments = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), size=(n_paths, steps))
    log_paths = np.cumsum(np.concatenate([np.zeros((n_paths,1)), increments], axis=1), axis=1)
    return S0 * np.exp(log_paths)

# ---------------------------------------------------------------------
# Data fetching 
# ---------------------------------------------------------------------
def fetch_history(ticker, period="2y"):
    """Return historical DataFrame (Close) for ticker. Handles errors."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

def fetch_risk_free_rate_yf():
    """
    Use yfinance to fetch a short-term Treasury yield proxy.
    ^IRX is the 13-week T-bill annualized yield in percent.
    We'll fetch the latest close and convert to decimal.
    If missing, return None.
    """
    try:
        r_df = yf.download("^IRX", period="7d", progress=False, auto_adjust=False)
        if r_df is None or r_df.empty:
            return None
        val = r_df["Close"].iloc[-1]
        return float(val) / 100.0
    except Exception:
        return None

def fetch_option_chain(ticker):
    """
    Fetch full option chain from yfinance.
    Returns dict: {expiry_str: DataFrame_with_calls_and_puts}
    Note: yfinance.Ticker.option_chain(date) returns a namedtuple(calls, puts)
    """
    t = yf.Ticker(ticker)
    try:
        expiries = t.options
    except Exception:
        expiries = []
    chains = {}
    for exp in expiries:
        try:
            oc = t.option_chain(exp)
            calls = oc.calls.copy()
            calls["expiry"] = pd.to_datetime(exp)
            calls["type"] = "call"
            puts = oc.puts.copy()
            puts["expiry"] = pd.to_datetime(exp)
            puts["type"] = "put"
            df = pd.concat([calls, puts], ignore_index=True, sort=False)
            chains[exp] = df
        except Exception:
            continue
    return chains

# ---------------------------------------------------------------------
# Strategy / Backtest 
# ---------------------------------------------------------------------
def model_price_series_for_option_history(S_series, K, days_to_expiry, r, method_vol="rolling", hv_window=30):
    """
    For each date t in S_series (index is dates), compute T (days_to_expiry),
    estimate historical vol using past hv_window days, then compute BS price.
    This function returns a series of theoretical option prices over the historical period.
    NOTE: This is model-based pricing (not market option history).
    """
    prices = []
    dates = S_series.index
    for i, date in enumerate(dates):
        if i < hv_window:
            prices.append(np.nan)
            continue
        past = S_series.iloc[max(0, i-hv_window):i+1]
        logr = np.log(past / past.shift(1)).dropna()
        if len(logr) < 5:
            prices.append(np.nan)
            continue
        sigma = np.sqrt(252) * logr.std()
        sigma = float(sigma)
        S0 = float(S_series.iloc[i])
        T = days_to_years(days_to_expiry)
        price = black_scholes_price(S0, K, T, r, sigma, "call")
        prices.append(price)
    return pd.Series(prices, index=dates)

def backtest_sell_monthly_atm(S_series, r, hv_window=30, trade_duration_days=30, notional=1.0):
    """
    Model-based backtest: every month (rolling, start-of-month), sell ATM call with X days to expiry (trade_duration_days).
    We price the short position at time t using model (historical vol up to t).
    Then at expiry (t + trade_duration_days), we compute realized P&L using actual underlying price.
    Returns DataFrame with trade records and cumulative equity.
    NOTE: this is synthetic (no historical option prices) and uses BS pricing for entry.
    """
    trades = []
    dates = S_series.index
    start_idx = hv_window
    i = start_idx
    while i < len(dates) - trade_duration_days:
        entry_date = dates[i]
        S_entry = float(S_series.loc[entry_date])
        K = round(S_entry, 2)
        past = S_series.iloc[max(0, i-hv_window):i+1]
        logr = np.log(past / past.shift(1)).dropna()
        if len(logr) < 5:
            i += 30
            continue
        sigma = float(np.sqrt(252) * logr.std())
        T = days_to_years(trade_duration_days)
        premium = black_scholes_price(S_entry, K, T, r, sigma, "call")
        expiry_date = dates[i + trade_duration_days]
        S_expiry = float(S_series.loc[expiry_date])
        payoff = min(0.0, premium - max(0.0, S_expiry - K)) 
        trades.append({
            "entry_date": entry_date,
            "expiry_date": expiry_date,
            "S_entry": S_entry,
            "S_expiry": S_expiry,
            "K": K,
            "sigma_used": sigma,
            "premium": premium,
            "pnl": payoff
        })
        i += 30  
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df["cum_pnl"] = df["pnl"].cumsum()
    df["return"] = df["pnl"] / df["premium"].replace(0, np.nan)
    return df

# ---------------------------------------------------------------------
# Streamlit UI 
# ---------------------------------------------------------------------
st.set_page_config(page_title="Black-Scholes Pro Dashboard", layout="wide")
st.title("📈 Black-Scholes Pro — Options Pricing, Heatmaps & Backtest")

st.sidebar.header("Inputs / Ticker")
ticker = st.sidebar.text_input("Ticker (yfinance)", "AAPL").strip().upper()
period_hist = st.sidebar.selectbox("Historical price window", ["1y", "2y", "5y"], index=1)
hv_window = st.sidebar.number_input("Historical vol window (days)", value=30, min_value=5, max_value=252)
trade_days = st.sidebar.number_input("Backtest trade duration (days)", value=30, min_value=1, max_value=365)
mc_paths = st.sidebar.number_input("Monte Carlo paths", value=5000, min_value=100, max_value=20000)
mc_steps = st.sidebar.number_input("Monte Carlo steps", value=252, min_value=10, max_value=2000)
notional = st.sidebar.number_input("Notional per option contract (for P&L scaling)", value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("OpenAI / Assistant")
use_ai = st.sidebar.checkbox("Enable GPT Assistant (gpt-4o-mini)", value=True)
if use_ai:
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_KEY:
        st.sidebar.error("Set OPENAI_API_KEY in .env to use assistant.")
    else:
        pass

st.sidebar.markdown("---")
st.sidebar.header("Quick Notes")
st.sidebar.write("""
- This app uses **model-based historical backtest** (recreating option price via Black-Scholes using historical realized vol).
- For *true historical option prices* you will need a paid data provider.
- IV is computed when the option's market price is available in the option chain (lastPrice or mid = (bid+ask)/2).
""")

st.markdown("## Market Data")
with st.spinner("Fetching historical prices and option chain..."):
    hist = fetch_history(ticker, period=period_hist)
    option_chains = fetch_option_chain(ticker)
    r_rf = fetch_risk_free_rate_yf()

if hist.empty:
    st.error("No historical price data found for that ticker. Check symbol and internet connection.")
    st.stop()

spot_price_raw = hist["Close"].iloc[-1]
if isinstance(spot_price_raw, (pd.Series, np.ndarray)):
    spot_price = float(np.asarray(spot_price_raw).squeeze())
else:
    spot_price = float(spot_price_raw)

st.write(f"**Ticker:** {ticker}  |  **Spot (last close):** {spot_price:.2f}")

if r_rf is None:
    st.warning("Automatic risk-free rate (IRX) not found. Please input manually.")
    rf_rate = st.number_input("Risk-free rate (decimal, e.g. 0.03)", value=0.02, format="%.4f")
else:
    rf_rate = st.number_input("Risk-free rate (decimal, e.g. 0.03)", value=float(r_rf), format="%.4f")
    st.caption(f"Auto-fetched short-term Treasury proxy (^IRX) = {r_rf*100:.3f}%")

logr = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
if len(logr) >= 5:
    hv = float(np.sqrt(252) * logr.rolling(hv_window).std().iloc[-1])
    st.markdown(f"**Historical Volatility ({hv_window}d, annualized):** {hv:.2%}")
else:
    hv = np.nan
    st.markdown("**Historical Volatility:** not enough history")

st.markdown("## Option Chain (current expiries)")
if not option_chains:
    st.warning("No option chain returned by yfinance. yfinance sometimes fails for some tickers or intraday frequency. The rest of the app will still work using model-based pricing.")
else:
    exp_summary = {k: len(v) for k, v in option_chains.items()}
    es = pd.DataFrame({"expiry": list(exp_summary.keys()), "rows": list(exp_summary.values())})
    st.dataframe(es)

st.markdown("## IV Surface & Greeks (from available option chain)")
iv_surface = None
greeks_table = None
if option_chains:
    rows = []
    for exp_str, df in option_chains.items():
        for _, r in df.iterrows():
            S = spot_price
            K = float(r.get("strike", r.get("Strike", np.nan)))
            expiry = pd.to_datetime(r["expiry"])
            days = (expiry.date() - datetime.today().date()).days
            T = max(days_to_years(days), 0.0001)
            market_price = None
            if pd.notna(r.get("lastPrice")) and r.get("lastPrice", 0) > 0:
                market_price = float(r["lastPrice"])
            else:
                bid = r.get("bid", np.nan)
                ask = r.get("ask", np.nan)
                if pd.notna(bid) and pd.notna(ask) and (bid > 0 or ask > 0):
                    market_price = float((float(bid) + float(ask)) / 2.0)
            option_type = r.get("type", "call")
            iv = np.nan
            if market_price is not None and market_price > 0:
                iv = implied_volatility(S, K, T, rf_rate, market_price, option_type)
            used_vol = iv if (not np.isnan(iv)) else (hv if not np.isnan(hv) else 0.2)
            delta, gamma, theta, vega, rho = option_greeks(S, K, T, rf_rate, used_vol, option_type)
            rows.append({
                "expiry": expiry,
                "days": days,
                "strike": K,
                "type": option_type,
                "market_price": market_price,
                "implied_vol": iv,
                "used_vol": used_vol,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho
            })
    if rows:
        ocdf = pd.DataFrame(rows)
        try:
            pivot_iv = ocdf.pivot_table(index="expiry", columns="strike", values="implied_vol")
            pivot_iv = pivot_iv.sort_index(ascending=True)
            iv_surface = pivot_iv
        except Exception:
            iv_surface = None
        greeks_table = ocdf

st.markdown("### Implied Volatility Surface (expiry × strike)")
if iv_surface is None or iv_surface.empty:
    st.info("No full IV surface available (option chain missing or implied vols not computable).")
else:
    iv_plot = iv_surface.copy()
    iv_plot = iv_plot.replace([np.inf, -np.inf], np.nan)
    fig_iv = go.Figure(data=go.Heatmap(
        z=iv_plot.values * 100.0, 
        x=list(map(str, iv_plot.columns)),
        y=[d.strftime("%Y-%m-%d") for d in iv_plot.index],
        colorbar=dict(title="IV (%)"),
        hovertemplate="strike=%{x}<br>expiry=%{y}<br>IV=%{z:.2f}%"
    ))
    fig_iv.update_layout(height=400, yaxis_nticks=10, xaxis_nticks=20)
    st.plotly_chart(fig_iv, use_container_width=True)

st.markdown("### Greeks Heatmap (Vega shown)")
if greeks_table is None or greeks_table.empty:
    st.info("Greeks not available (no option chain or no computed rows).")
else:
    try:
        vega_pivot = greeks_table.pivot_table(index="expiry", columns="strike", values="vega")
        vega_plot = vega_pivot.copy()
        fig_vega = go.Figure(data=go.Heatmap(
            z=vega_plot.values,
            x=list(map(str, vega_plot.columns)),
            y=[d.strftime("%Y-%m-%d") for d in vega_plot.index],
            colorbar=dict(title="Vega (per 1% vol)"),
            hovertemplate="strike=%{x}<br>expiry=%{y}<br>vega=%{z:.4f}"
        ))
        fig_vega.update_layout(height=400)
        st.plotly_chart(fig_vega, use_container_width=True)
    except Exception:
        st.info("Unable to pivot greeks for heatmap.")

st.markdown("### Short Call P&L Heatmap (selected expiry & strike)")
if greeks_table is None or greeks_table.empty:
    st.info("No option selected — P&L heatmap requires option chain. You can still use Monte Carlo and model backtest.")
else:
    expiries = sorted(greeks_table["expiry"].dt.strftime("%Y-%m-%d").unique())
    sel_exp = st.selectbox("Select expiry", expiries, index=0)
    strikes = sorted(greeks_table[greeks_table["expiry"].dt.strftime("%Y-%m-%d") == sel_exp]["strike"].unique())
    sel_strike = st.selectbox("Select strike", strikes, index=0)
    sel_type = st.selectbox("Select type", ["call", "put"], index=0)
    selected_row = greeks_table[(greeks_table["expiry"].dt.strftime("%Y-%m-%d")==sel_exp) & (greeks_table["strike"]==sel_strike) & (greeks_table["type"]==sel_type)]
    if selected_row.empty:
        st.info("Selected option not found in chain.")
    else:
        sr = selected_row.iloc[0]
        market_price = sr["market_price"] if not pd.isna(sr["market_price"]) else 0.0
        used_iv = sr["used_vol"]
        days = int(sr["days"])
        T = max(days_to_years(days), 0.0001)
        S_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 80)
        vol_range = np.linspace(max(0.01, used_iv * 0.5), used_iv * 1.6, 80)
        pnl = np.zeros((len(S_range), len(vol_range)))
        for i, S in enumerate(S_range):
            for j, vol in enumerate(vol_range):
                the_price = black_scholes_price(S, sel_strike, T, rf_rate, vol, sel_type)
                if market_price > 0:
                    pnl[i, j] = market_price - the_price 
                else:
                    pnl[i, j] = -the_price  
        fig_pnl = go.Figure(data=go.Heatmap(
            z=pnl,
            x=np.round(vol_range, 4),
            y=np.round(S_range, 2),
            colorbar=dict(title="P&L (short)"),
            hovertemplate="Vol=%{x}<br>S=%{y}<br>P&L=%{z:.4f}"
        ))
        fig_pnl.update_layout(height=450, xaxis_title="Volatility", yaxis_title="Stock Price")
        st.plotly_chart(fig_pnl, use_container_width=True)

# ---------------------------------------------------------------------
# Monte Carlo Simulation panel
# ---------------------------------------------------------------------
st.markdown("## Monte Carlo Simulation (distribution of payoff)")

st.markdown("Choose option to simulate (or simulate ATM call with historical vol):")
mc_use_chain = st.checkbox("Simulate selected option from chain", value=True if (greeks_table is not None and not greeks_table.empty) else False)
mc_trials = int(mc_paths)
mc_steps = int(mc_steps)

if mc_use_chain and not (greeks_table is not None and not greeks_table.empty):
    st.warning("No chain available — switching to model ATM simulation.")
    mc_use_chain = False

if mc_use_chain:
    base_sigma = sr["used_vol"]
    K_mc = sel_strike
    T_mc = max(days_to_years(int(sr["days"])), 0.0001)
    S0_mc = spot_price
    st.write(f"Simulating option {ticker} {sel_strike} {sel_exp} ({sel_type}). Using vol={base_sigma:.2%}, T={T_mc:.3f}y")
else:
    K_mc = round(spot_price, 2)
    T_mc = days_to_years(trade_days)
    base_sigma = hv if not np.isnan(hv) else 0.2
    S0_mc = spot_price
    st.write(f"Simulating ATM call K={K_mc}, T={T_mc:.3f}y, vol={base_sigma:.2%}")

mc_seed = 42
with st.spinner("Running Monte Carlo..."):
    paths = gbm_simulate_paths(S0_mc, mu=0.0, sigma=base_sigma, T_years=T_mc, steps=mc_steps, n_paths=mc_trials, seed=mc_seed)
    ST = paths[:, -1]
    premium_mc = black_scholes_price(S0_mc, K_mc, T_mc, rf_rate, base_sigma, "call")
    payoff_short = premium_mc - np.maximum(ST - K_mc, 0)
    prob_profit = np.mean(payoff_short > 0)
    expected_pnl = payoff_short.mean()
    st.metric("Monte Carlo: Prob(Short Call Profit)", f"{prob_profit:.2%}", delta=None)
    st.metric("Monte Carlo: Expected P&L (per option)", f"{expected_pnl:.4f}")
    fig_mc = px.histogram(payoff_short, nbins=80, title="Monte Carlo distribution of short-call P&L")
    st.plotly_chart(fig_mc, use_container_width=True)

# ---------------------------------------------------------------------
# Historical backtest 
# ---------------------------------------------------------------------
st.markdown("## Historical Backtest (model-based: monthly sell ATM calls)")
with st.spinner("Running historical backtest..."):
    s_series = hist["Close"]
    bt_df = backtest_sell_monthly_atm(s_series, rf_rate, hv_window=hv_window, trade_duration_days=trade_days)
if bt_df.empty:
    st.info("Backtest returned no trades (not enough history).")
else:
    st.write("Backtest trade records (sample):")
    st.dataframe(bt_df.head(30))
    fig_bt = px.line(bt_df, x="expiry_date", y="cum_pnl", title="Cumulative P&L from model-based monthly short-call strategy")
    st.plotly_chart(fig_bt, use_container_width=True)
    total_pnl = bt_df["pnl"].sum()
    win_rate = np.mean(bt_df["pnl"] > 0)
    st.markdown(f"**Total P&L:** {total_pnl:.4f}  &nbsp;&nbsp; **Win rate:** {win_rate:.2%}  &nbsp;&nbsp; **Trades:** {len(bt_df)}")

# ---------------------------------------------------------------------
# GPT-based Assistant 
# ---------------------------------------------------------------------
st.markdown("## GPT Assistant (explainers & guidance)")
if use_ai and OPENAI_KEY:
    if OpenAI is None:
        st.error("OpenAI SDK not available in your environment. Install the python client (openai).")
    else:
        client = OpenAI(api_key=OPENAI_KEY)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for m in st.session_state.chat_history:
            role = m["role"]
            content = m["content"]
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Assistant:** {content}")
        user_msg = st.text_input("Ask the quant assistant about results, IV, backtest, or risk-free rate...", "")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            context = f"""
You are a quant assistant. The user is looking at ticker {ticker}. Spot={spot_price:.2f}, historical vol={hv:.2%}.
We pulled option chain expiries: {list(option_chains.keys())} (count={len(option_chains)}).
Risk-free rate used: {rf_rate:.4f}.
We computed an IV surface (if available). Backtest: selling ATM short monthly for duration {trade_days} days.
"""
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a concise quant assistant. Give precise, practical answers and explain assumptions."},
                        {"role": "user", "content": context + "\nUser question:\n" + user_msg}
                    ],
                    max_tokens=400
                )
                assistant_text = resp.choices[0].message.content
            except Exception as e:
                assistant_text = f"Assistant error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
            st.markdown(f"**Assistant:** {assistant_text}")
else:
    st.info("GPT Assistant disabled or API key missing. Use the sidebar to enable and set OPENAI_API_KEY in .env.")

# ---------------------------------------------------------------------
# Disclaimers
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("""
**Disclaimers & Notes**

- This tool is for educational and decision-support use. It is *not* trading advice.
- IVs are computed when market option price exists. yfinance option chain may not always return last prices for all strikes.
- Historical backtest here is **model-based** (we reconstruct option prices using Black–Scholes and historical realized volatility). For *true* historical option P/L, use a paid historical options dataset.
- Monte Carlo uses Geometric Brownian Motion — you should calibrate drift/vol and consider more advanced models (stochastic vol, jumps) for production usage.
""")
