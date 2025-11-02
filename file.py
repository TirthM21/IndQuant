# streamlit_portfolio_backtest_india_advanced.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Advanced India Portfolio Backtester")

# --------------------------
# Expanded Indian Universe
# --------------------------
INDIAN_INDICES = {
    "NIFTY 50": "^NSEI",
    "BSE SENSEX": "^BSESN",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY Midcap 100": "^NSEMDCP100",
    "NIFTY Next 50": "^NSMIDCP"
}

INDIAN_STOCKS = {
    # IT Sector
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HCL Tech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    
    # Banking & Finance
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    
    # Energy & Oil
    "Reliance Industries": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "Adani Green": "ADANIGREEN.NS",
    
    # Telecom
    "Bharti Airtel": "BHARTIARTL.NS",
    "Vodafone Idea": "IDEA.NS",
    
    # Auto
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    
    # Consumer
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Titan Company": "TITAN.NS",
    "Nestle India": "NESTLEIND.NS",
    
    # Infrastructure
    "Larsen & Toubro": "LT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Adani Ports": "ADANIPORTS.NS",
    
    # Pharma
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Labs": "DIVISLAB.NS"
}

UNIVERSE = {**INDIAN_INDICES, **INDIAN_STOCKS}

# --------------------------
# Advanced Helper Functions
# --------------------------
@st.cache_data(ttl=3600)
def download_multi_tickers(tickers, start, end):
    """Download with better error handling"""
    try:
        df = yf.download(tickers, start=start, end=end, auto_adjust=True, 
                        threads=True, group_by='ticker', progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            cols = {}
            for ticker in tickers:
                if ticker in df.columns.levels[0]:
                    cols[ticker] = df[ticker]['Close']
                elif ('Close', ticker) in df.columns:
                    cols[ticker] = df[('Close', ticker)]
            adj_close = pd.DataFrame(cols)
        else:
            adj_close = df['Close'] if 'Close' in df.columns else df
        
        adj_close.index = pd.to_datetime(adj_close.index)
        return adj_close.sort_index()
    except Exception as e:
        st.error(f"Download error: {e}")
        return pd.DataFrame()

def compute_returns(price_df):
    return price_df.pct_change().dropna(how='all')

def CAGR(returns, periods_per_year=252):
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    return cumulative ** (1 / n_years) - 1 if n_years > 0 else 0.0

def annualized_vol(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def max_drawdown(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    return drawdown.min()

def sharpe_ratio(returns, rf=0.06, periods_per_year=252):
    """Sharpe ratio with Indian risk-free rate default"""
    excess = returns - rf / periods_per_year
    ann_ret = CAGR(excess, periods_per_year)
    ann_vol = annualized_vol(returns, periods_per_year)
    return ann_ret / ann_vol if ann_vol > 0 else 0.0

def sortino_ratio(returns, rf=0.06, periods_per_year=252):
    """Sortino ratio - downside deviation only"""
    excess = returns - rf / periods_per_year
    ann_ret = CAGR(excess, periods_per_year)
    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    return ann_ret / downside if downside > 0 else 0.0

def calmar_ratio(returns, cum_returns):
    """Calmar ratio - CAGR / Max Drawdown"""
    cagr = CAGR(returns)
    mdd = abs(max_drawdown(cum_returns))
    return cagr / mdd if mdd > 0 else 0.0

def omega_ratio(returns, threshold=0.0):
    """Omega ratio - probability weighted gains vs losses"""
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    return gains / losses if losses > 0 else 0.0

def historical_var(returns, p=0.95):
    return -np.percentile(returns.dropna(), (1-p)*100) if len(returns.dropna())>0 else 0.0

def conditional_var(returns, p=0.95):
    """CVaR / Expected Shortfall"""
    var = historical_var(returns, p)
    return -returns[returns <= -var].mean() if len(returns[returns <= -var]) > 0 else 0.0

def rolling_sharpe(returns, window=252, periods_per_year=252):
    """Rolling Sharpe ratio"""
    rolling_ret = returns.rolling(window).mean() * periods_per_year
    rolling_vol = returns.rolling(window).std() * np.sqrt(periods_per_year)
    return rolling_ret / rolling_vol

def beta_alpha(returns, benchmark_returns):
    """Calculate beta and alpha vs benchmark"""
    covariance = returns.cov(benchmark_returns)
    benchmark_var = benchmark_returns.var()
    beta = covariance / benchmark_var if benchmark_var > 0 else 0.0
    
    port_return = returns.mean() * 252
    bench_return = benchmark_returns.mean() * 252
    alpha = port_return - beta * bench_return
    
    return beta, alpha

# --------------------------
# Advanced Strategy Functions
# --------------------------
def buy_and_hold_signal(price):
    return pd.Series(1, index=price.index)

def sma_crossover_signals(price, short=20, long=50):
    sma_short = price.rolling(short).mean()
    sma_long = price.rolling(long).mean()
    signal = (sma_short > sma_long).astype(int)
    return signal.shift(1).fillna(0)

def rsi(price, window=14):
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def rsi_signals(price, low=30, high=70):
    r = rsi(price)
    sig = pd.Series(0, index=price.index)
    sig[r < low] = 1
    sig[r > high] = 0
    return sig.ffill().fillna(0)

def bollinger_bands_signals(price, window=20, num_std=2):
    """Bollinger Bands strategy"""
    sma = price.rolling(window).mean()
    std = price.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    
    sig = pd.Series(0, index=price.index)
    sig[price < lower] = 1  # Buy when price below lower band
    sig[price > upper] = 0   # Sell when price above upper band
    return sig.ffill().fillna(0)

def macd_signals(price, fast=12, slow=26, signal=9):
    """MACD strategy"""
    ema_fast = price.ewm(span=fast).mean()
    ema_slow = price.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    
    sig = (macd_line > signal_line).astype(int)
    return sig.shift(1).fillna(0)

def momentum_signals(price, window=90, threshold=0.0):
    """Momentum strategy - buy if return > threshold"""
    momentum = price / price.shift(window) - 1
    sig = (momentum > threshold).astype(int)
    return sig.shift(1).fillna(0)

def mean_reversion_signals(price, window=20, z_threshold=2):
    """Mean reversion strategy"""
    sma = price.rolling(window).mean()
    std = price.rolling(window).std()
    z_score = (price - sma) / std
    
    sig = pd.Series(0, index=price.index)
    sig[z_score < -z_threshold] = 1  # Buy when oversold
    sig[z_score > z_threshold] = 0    # Sell when overbought
    return sig.ffill().fillna(0)

# --------------------------
# Portfolio Optimization Functions
# --------------------------
def equal_weight(n_assets):
    return np.ones(n_assets) / n_assets

def inverse_volatility_weight(returns):
    """Weight inversely proportional to volatility"""
    vols = returns.std()
    inv_vols = 1 / vols
    return (inv_vols / inv_vols.sum()).values

def risk_parity_weight(returns):
    """Risk parity - equal risk contribution"""
    cov = returns.cov().values
    n = len(returns.columns)
    
    def risk_contribution(w):
        portfolio_vol = np.sqrt(w @ cov @ w)
        marginal_contrib = cov @ w
        contrib = w * marginal_contrib / portfolio_vol
        return contrib
    
    def objective(w):
        contrib = risk_contribution(w)
        target = np.ones(n) / n
        return ((contrib - target) ** 2).sum()
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    x0 = equal_weight(n)
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else x0

def max_sharpe_weight(returns, rf=0.06):
    """Maximum Sharpe ratio portfolio"""
    mean_returns = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(returns.columns)
    
    def neg_sharpe(w):
        ret = w @ mean_returns
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol if vol > 0 else 0
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    x0 = equal_weight(n)
    
    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else x0

def min_variance_weight(returns):
    """Minimum variance portfolio"""
    cov = returns.cov().values
    n = len(returns.columns)
    
    def portfolio_variance(w):
        return w @ cov @ w
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    x0 = equal_weight(n)
    
    result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else x0

# --------------------------
# Streamlit UI
# --------------------------
st.title("🇮🇳 Advanced India Portfolio Backtester")
st.markdown("*Professional-grade backtesting with advanced strategies, optimization, and risk analytics*")

# Sidebar
with st.sidebar:
    st.header("📊 Asset Selection")
    
    # Sector filter
    sector_filter = st.multiselect(
        "Filter by sector",
        ["All", "IT", "Banking", "Energy", "Auto", "Consumer", "Pharma", "Infrastructure"],
        default=["All"]
    )
    
    # Smart filtering
    if "All" not in sector_filter and sector_filter:
        filtered_stocks = {}
        sector_map = {
            "IT": ["TCS", "Infosys", "HCL Tech", "Wipro", "Tech Mahindra"],
            "Banking": ["HDFC Bank", "ICICI Bank", "Kotak Mahindra Bank", "State Bank of India", "Axis Bank"],
            "Energy": ["Reliance Industries", "ONGC", "NTPC", "Power Grid"],
            "Auto": ["Maruti Suzuki", "Tata Motors", "Mahindra & Mahindra", "Bajaj Auto"],
            "Consumer": ["Hindustan Unilever", "ITC", "Asian Paints", "Titan Company"],
            "Pharma": ["Sun Pharma", "Dr. Reddy's", "Cipla", "Divi's Labs"],
            "Infrastructure": ["Larsen & Toubro", "UltraTech Cement", "Adani Ports"]
        }
        for sector in sector_filter:
            for stock in sector_map.get(sector, []):
                if stock in INDIAN_STOCKS:
                    filtered_stocks[stock] = INDIAN_STOCKS[stock]
        available_universe = {**INDIAN_INDICES, **filtered_stocks}
    else:
        available_universe = UNIVERSE
    
    options = list(available_universe.keys())
    selected_labels = st.multiselect(
        "Select assets", 
        options=options, 
        default=["NIFTY 50", "Reliance Industries", "HDFC Bank", "TCS"]
    )
    selected_tickers = [available_universe[label] for label in selected_labels]
    
    # Benchmark selection
    st.markdown("---")
    st.header("📈 Benchmark")
    benchmark_label = st.selectbox("Benchmark index", list(INDIAN_INDICES.keys()), index=0)
    benchmark_ticker = INDIAN_INDICES[benchmark_label]
    
    st.markdown("---")
    st.header("📅 Backtest Period")
    period_preset = st.selectbox("Quick select", ["Custom", "1 Year", "3 Years", "5 Years", "10 Years", "Max"])
    
    if period_preset == "Custom":
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=365*3))
        end_date = st.date_input("End date", value=date.today())
    else:
        end_date = date.today()
        period_map = {"1 Year": 365, "3 Years": 365*3, "5 Years": 365*5, "10 Years": 365*10, "Max": 365*20}
        start_date = end_date - timedelta(days=period_map[period_preset])
    
    st.markdown("---")
    st.header("🎯 Strategy")
    strategy_choice = st.selectbox(
        "Select strategy",
        ["Buy & Hold", "SMA Crossover", "RSI-based", "Bollinger Bands", 
         "MACD", "Momentum", "Mean Reversion"]
    )
    
    # Strategy parameters
    if strategy_choice == "SMA Crossover":
        col1, col2 = st.columns(2)
        sma_short = col1.number_input("Short period", value=20, min_value=5)
        sma_long = col2.number_input("Long period", value=50, min_value=10)
    elif strategy_choice == "RSI-based":
        col1, col2 = st.columns(2)
        rsi_low = col1.number_input("Oversold", value=30, min_value=0, max_value=50)
        rsi_high = col2.number_input("Overbought", value=70, min_value=50, max_value=100)
    elif strategy_choice == "Momentum":
        momentum_window = st.number_input("Lookback period", value=90, min_value=20)
    
    st.markdown("---")
    st.header("⚖️ Portfolio Weighting")
    weight_method = st.selectbox(
        "Weighting scheme",
        ["Equal Weight", "Inverse Volatility", "Risk Parity", "Max Sharpe", "Min Variance"]
    )
    
    rebalance_freq = st.selectbox("Rebalance frequency", ["Daily", "Weekly", "Monthly", "Quarterly", "Annual"])
    
    st.markdown("---")
    st.header("💰 Capital & Costs")
    initial_capital = st.number_input("Initial capital (₹)", value=10000000, step=100000, format="%d")
    transaction_cost = st.number_input("Transaction cost (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.01) / 100
    slippage = st.number_input("Slippage (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100
    
    st.markdown("---")
    st.header("⚙️ Advanced Options")
    risk_free_rate = st.number_input("Risk-free rate (%)", value=6.5, step=0.1) / 100
    confidence_level = st.slider("VaR confidence level", 0.90, 0.99, 0.95, 0.01)

# Main content
if len(selected_tickers) == 0:
    st.warning("⚠️ Please select at least one asset from the sidebar.")
    st.stop()

# Download data
with st.spinner("📥 Downloading market data..."):
    prices = download_multi_tickers(selected_tickers + [benchmark_ticker], start_date, end_date)
    
if prices.empty:
    st.error("❌ No data available for selected tickers and date range.")
    st.stop()

# Separate benchmark
benchmark_prices = prices[benchmark_ticker] if benchmark_ticker in prices.columns else None
prices = prices[[c for c in prices.columns if c in selected_tickers]]

st.success(f"✅ Downloaded {prices.shape[0]} days of data for {prices.shape[1]} assets")

# Compute returns
returns = compute_returns(prices)
benchmark_returns = compute_returns(benchmark_prices) if benchmark_prices is not None else None

# Generate signals based on strategy
strategy_map = {
    "Buy & Hold": lambda p: buy_and_hold_signal(p),
    "SMA Crossover": lambda p: sma_crossover_signals(p, sma_short, sma_long),
    "RSI-based": lambda p: rsi_signals(p, rsi_low, rsi_high),
    "Bollinger Bands": lambda p: bollinger_bands_signals(p),
    "MACD": lambda p: macd_signals(p),
    "Momentum": lambda p: momentum_signals(p, momentum_window),
    "Mean Reversion": lambda p: mean_reversion_signals(p)
}

signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
for col in prices.columns:
    try:
        signals[col] = strategy_map[strategy_choice](prices[col])
    except:
        signals[col] = 1

# Apply weighting scheme
weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

# Get rebalance dates
if rebalance_freq == "Daily":
    rebal_dates = prices.index
elif rebalance_freq == "Weekly":
    rebal_dates = prices.resample('W').last().index
elif rebalance_freq == "Monthly":
    rebal_dates = prices.resample('M').last().index
elif rebalance_freq == "Quarterly":
    rebal_dates = prices.resample('Q').last().index
else:  # Annual
    rebal_dates = prices.resample('Y').last().index

# Calculate weights at rebalance dates
weight_functions = {
    "Equal Weight": lambda r: equal_weight(len(r.columns)),
    "Inverse Volatility": inverse_volatility_weight,
    "Risk Parity": risk_parity_weight,
    "Max Sharpe": lambda r: max_sharpe_weight(r, risk_free_rate),
    "Min Variance": min_variance_weight
}

last_weights = None
lookback_period = 252  # 1 year for weight calculation

for dt in prices.index:
    if dt in rebal_dates:
        # Get active signals
        active_cols = signals.loc[dt][signals.loc[dt] == 1].index.tolist()
        
        if len(active_cols) > 0:
            # Get historical returns for weight calculation
            hist_returns = returns.loc[:dt, active_cols].tail(lookback_period)
            
            if len(hist_returns) > 20:  # Minimum data requirement
                try:
                    w = weight_functions[weight_method](hist_returns)
                    for i, col in enumerate(active_cols):
                        weights.loc[dt, col] = w[i]
                    last_weights = weights.loc[dt]
                except:
                    # Fallback to equal weight
                    for col in active_cols:
                        weights.loc[dt, col] = 1.0 / len(active_cols)
                    last_weights = weights.loc[dt]
    elif last_weights is not None:
        weights.loc[dt] = last_weights

# Calculate portfolio returns
port_returns = (weights.shift(1) * returns).sum(axis=1)

# Apply transaction costs and slippage
position_changes = weights.diff().abs().sum(axis=1)
total_costs = position_changes * (transaction_cost + slippage)
port_returns = port_returns - total_costs

# Calculate cumulative returns
cum_returns = (1 + port_returns).cumprod()
cum_benchmark = (1 + benchmark_returns).cumprod() if benchmark_returns is not None else None

# --------------------------
# Display Results
# --------------------------
st.header("📊 Performance Dashboard")

# Key metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    cagr_val = CAGR(port_returns.dropna())
    st.metric("CAGR", f"{cagr_val:.2%}")

with col2:
    vol_val = annualized_vol(port_returns.dropna())
    st.metric("Volatility", f"{vol_val:.2%}")

with col3:
    sharpe_val = sharpe_ratio(port_returns.dropna(), risk_free_rate)
    st.metric("Sharpe Ratio", f"{sharpe_val:.2f}")

with col4:
    mdd_val = max_drawdown(cum_returns)
    st.metric("Max Drawdown", f"{mdd_val:.2%}")

with col5:
    final_value = initial_capital * cum_returns.iloc[-1]
    st.metric("Final Value", f"₹{final_value:,.0f}")

# Charts
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📉 Risk Analysis", "💼 Portfolio", "📊 Statistics"])

with tab1:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Cumulative Returns", "Drawdown", "Rolling Sharpe (1Y)")
    )
    
    # Cumulative returns
    fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns.values, 
                            name='Portfolio', line=dict(color='blue', width=2)), row=1, col=1)
    if cum_benchmark is not None:
        fig.add_trace(go.Scatter(x=cum_benchmark.index, y=cum_benchmark.values,
                                name='Benchmark', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    
    # Drawdown
    drawdown = cum_returns / cum_returns.cummax() - 1
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, 
                            name='Drawdown', fill='tozeroy', line=dict(color='red')), row=2, col=1)
    
    # Rolling Sharpe
    roll_sharpe = rolling_sharpe(port_returns, window=252)
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                            name='Rolling Sharpe', line=dict(color='green')), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
    st.plotly_chart(fig, width='stretch')
    
    # Monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")
    monthly_returns = port_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_pivot = monthly_returns.to_frame('Returns')
    monthly_pivot['Year'] = monthly_pivot.index.year
    monthly_pivot['Month'] = monthly_pivot.index.month
    pivot_table = monthly_pivot.pivot(index='Year', columns='Month', values='Returns')
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_table.values * 100,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot_table.index,
        colorscale='RdYlGn',
        text=np.round(pivot_table.values * 100, 2),
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Return %")
    ))
    fig_heatmap.update_layout(height=400, title="Monthly Returns (%)")
    st.plotly_chart(fig_heatmap, width='stretch')

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Metrics")
        risk_metrics = {
            "Volatility (Ann.)": f"{annualized_vol(port_returns.dropna()):.2%}",
            "Downside Deviation": f"{port_returns[port_returns < 0].std() * np.sqrt(252):.2%}",
            "Sortino Ratio": f"{sortino_ratio(port_returns.dropna(), risk_free_rate):.2f}",
            "Calmar Ratio": f"{calmar_ratio(port_returns.dropna(), cum_returns):.2f}",
            "Omega Ratio": f"{omega_ratio(port_returns.dropna()):.2f}",
            f"VaR ({confidence_level:.0%})": f"{historical_var(port_returns.dropna(), confidence_level):.2%}",
            f"CVaR ({confidence_level:.0%})": f"{conditional_var(port_returns.dropna(), confidence_level):.2%}",
            "Skewness": f"{port_returns.dropna().skew():.2f}",
            "Kurtosis": f"{port_returns.dropna().kurtosis():.2f}"
        }
        st.dataframe(pd.DataFrame.from_dict(risk_metrics, orient='index', columns=['Value']), width='stretch')
    
    with col2:
        if benchmark_returns is not None:
            st.subheader("vs Benchmark")
            
            # --- FIX STARTS HERE ---
            # To fix the 'Unalignable boolean Series' error, we combine portfolio and benchmark returns
            # into a single DataFrame. This automatically aligns them by their index.
            combined_df = pd.DataFrame({'portfolio': port_returns, 'benchmark': benchmark_returns}).dropna()
            
            port_returns_aligned = combined_df['portfolio']
            benchmark_returns_aligned = combined_df['benchmark']

            beta, alpha = beta_alpha(port_returns_aligned, benchmark_returns_aligned)

            # Calculate Up and Down Capture Ratios safely
            up_mask = benchmark_returns_aligned > 0
            down_mask = benchmark_returns_aligned < 0
            
            port_up_mean = port_returns_aligned[up_mask].mean()
            bench_up_mean = benchmark_returns_aligned[up_mask].mean()
            
            port_down_mean = port_returns_aligned[down_mask].mean()
            bench_down_mean = benchmark_returns_aligned[down_mask].mean()

            up_capture = port_up_mean / bench_up_mean if bench_up_mean != 0 else 0
            down_capture = port_down_mean / bench_down_mean if bench_down_mean != 0 else 0
            # --- FIX ENDS HERE ---

            bench_metrics = {
                "Beta": f"{beta:.2f}",
                "Alpha (Ann.)": f"{alpha:.2%}",
                "Correlation": f"{port_returns_aligned.corr(benchmark_returns_aligned):.2f}",
                "Tracking Error": f"{(port_returns_aligned - benchmark_returns_aligned).std() * np.sqrt(252):.2%}",
                "Information Ratio": f"{(CAGR(port_returns_aligned) - CAGR(benchmark_returns_aligned)) / ((port_returns_aligned - benchmark_returns_aligned).std() * np.sqrt(252)):.2f}",
                "Up Capture": f"{up_capture:.2%}",
                "Down Capture": f"{down_capture:.2%}"
            }
            st.dataframe(pd.DataFrame.from_dict(bench_metrics, orient='index', columns=['Value']), width='stretch')
    
    # Return distribution
    st.subheader("Return Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=port_returns.dropna() * 100, nbinsx=50, name='Portfolio Returns', 
                                    marker_color='blue', opacity=0.7))
    fig_dist.update_layout(height=400, xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                          title="Distribution of Daily Returns")
    st.plotly_chart(fig_dist, width='stretch')
    
    # Rolling volatility
    st.subheader("Rolling Volatility (30-day)")
    rolling_vol = port_returns.rolling(30).std() * np.sqrt(252)
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, 
                                 line=dict(color='orange'), name='30-day Rolling Vol'))
    fig_vol.update_layout(height=300, yaxis_title="Annualized Volatility")
    st.plotly_chart(fig_vol, width='stretch')

with tab3:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio evolution over time
        st.subheader("Portfolio Weights Evolution")
        fig_weights = go.Figure()
        for col in weights.columns:
            fig_weights.add_trace(go.Scatter(x=weights.index, y=weights[col], 
                                            name=col, stackgroup='one', mode='none'))
        fig_weights.update_layout(height=400, yaxis_title="Weight", 
                                 title=f"Asset Allocation Over Time ({weight_method})")
        st.plotly_chart(fig_weights, width='stretch')
    
    with col2:
        # Current allocation
        st.subheader("Current Allocation")
        latest_weights = weights.iloc[-1]
        latest_weights = latest_weights[latest_weights > 0.001]
        if not latest_weights.empty:
            fig_pie = go.Figure(data=[go.Pie(labels=latest_weights.index, 
                                             values=latest_weights.values,
                                             hole=0.4)])
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, width='stretch')
    
    # Turnover analysis
    st.subheader("Portfolio Turnover")
    turnover = weights.diff().abs().sum(axis=1).cumsum()
    avg_monthly_turnover = weights.diff().abs().sum(axis=1).resample('M').mean().mean()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_turn = go.Figure()
        fig_turn.add_trace(go.Scatter(x=turnover.index, y=turnover.values, 
                                      line=dict(color='purple')))
        fig_turn.update_layout(height=300, title="Cumulative Turnover", 
                              yaxis_title="Turnover")
        st.plotly_chart(fig_turn, width='stretch')
    
    with col2:
        st.metric("Avg Monthly Turnover", f"{avg_monthly_turnover:.2%}")
        st.metric("Total Turnover", f"{turnover.iloc[-1]:.2f}x")
        total_transaction_costs = (transaction_cost + slippage) * turnover.iloc[-1] * initial_capital
        st.metric("Total Transaction Costs", f"₹{total_transaction_costs:,.0f}")

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Individual Asset Performance")
        asset_stats = []
        for ticker in prices.columns:
            r = returns[ticker].dropna()
            if len(r) > 5:
                asset_stats.append({
                    "Asset": ticker,
                    "CAGR": f"{CAGR(r):.2%}",
                    "Volatility": f"{annualized_vol(r):.2%}",
                    "Sharpe": f"{sharpe_ratio(r, risk_free_rate):.2f}",
                    "Max DD": f"{max_drawdown((1+r).cumprod()):.2%}",
                    "Avg Weight": f"{weights[ticker].mean():.2%}"
                })
        if asset_stats:
            asset_df = pd.DataFrame(asset_stats)
            st.dataframe(asset_df, width='stretch', hide_index=True)
    
    with col2:
        st.subheader("Period Analysis")
        
        # Best/Worst periods
        best_day = port_returns.idxmax()
        worst_day = port_returns.idxmin()
        best_month = monthly_returns.idxmax()
        worst_month = monthly_returns.idxmin()
        
        period_stats = {
            "Best Day": f"{port_returns.max():.2%} ({best_day.strftime('%Y-%m-%d')})",
            "Worst Day": f"{port_returns.min():.2%} ({worst_day.strftime('%Y-%m-%d')})",
            "Best Month": f"{monthly_returns.max():.2%} ({best_month.strftime('%Y-%m')})",
            "Worst Month": f"{monthly_returns.min():.2%} ({worst_month.strftime('%Y-%m')})",
            "Positive Days": f"{(port_returns > 0).sum() / len(port_returns):.1%}",
            "Positive Months": f"{(monthly_returns > 0).sum() / len(monthly_returns):.1%}"
        }
        st.dataframe(pd.DataFrame.from_dict(period_stats, orient='index', columns=['Value']), 
                    width='stretch')
    
    # Correlation matrix
    st.subheader("Asset Correlation Matrix")
    corr_matrix = returns[prices.columns].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig_corr.update_layout(height=500, title="Correlation Matrix")
    st.plotly_chart(fig_corr, width='stretch')
    
    # Rolling correlation with benchmark
    if benchmark_returns is not None:
        st.subheader("Rolling Correlation with Benchmark (90-day)")
        rolling_corr = port_returns.rolling(90).corr(benchmark_returns)
        fig_rcorr = go.Figure()
        fig_rcorr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values,
                                       line=dict(color='teal')))
        fig_rcorr.update_layout(height=300, yaxis_title="Correlation")
        st.plotly_chart(fig_rcorr, width='stretch')

# --------------------------
# Advanced Analytics Section
# --------------------------
st.header("🔬 Advanced Analytics")

tab_adv1, tab_adv2, tab_adv3 = st.tabs(["📉 Scenario Analysis", "🎲 Monte Carlo", "📄 Reports"])

with tab_adv1:
    st.subheader("Stress Testing Scenarios")
    
    # Historical scenarios
    scenarios = {
        "COVID Crash (Mar 2020)": (pd.Timestamp('2020-02-20'), pd.Timestamp('2020-03-24')),
        "2018 Correction": (pd.Timestamp('2018-08-01'), pd.Timestamp('2018-10-31')),
        "Demonetization (Nov 2016)": (pd.Timestamp('2016-11-01'), pd.Timestamp('2016-11-30'))
    }
    
    scenario_results = []
    for scenario_name, (start, end) in scenarios.items():
        mask = (port_returns.index >= start) & (port_returns.index <= end)
        if mask.sum() > 0:
            scenario_ret = (1 + port_returns[mask]).prod() - 1
            scenario_results.append({
                "Scenario": scenario_name,
                "Return": f"{scenario_ret:.2%}",
                "Days": mask.sum()
            })
    
    if scenario_results:
        st.dataframe(pd.DataFrame(scenario_results), width='stretch', hide_index=True)
    
    # Custom scenario
    st.subheader("Custom Stress Test")
    col1, col2, col3 = st.columns(3)
    with col1:
        shock_size = st.slider("Market shock (%)", -50, -5, -20)
    with col2:
        shock_vol_mult = st.slider("Volatility multiplier", 1.0, 3.0, 1.5)
    with col3:
        recovery_days = st.slider("Recovery period (days)", 10, 180, 60)
    
    if st.button("Run Stress Test"):
        # Simulate shock scenario
        current_value = initial_capital * cum_returns.iloc[-1]
        shocked_value = current_value * (1 + shock_size/100)
        days_to_recover = len([i for i in range(len(cum_returns)-1, 0, -1) 
                              if cum_returns.iloc[i] * initial_capital >= shocked_value])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Portfolio Loss", f"₹{current_value - shocked_value:,.0f}")
        col2.metric("Historical Recovery Time", f"{days_to_recover} days")
        col3.metric("Estimated New Recovery", f"~{recovery_days} days")

with tab_adv2:
    st.subheader("Monte Carlo Simulation")
    
    n_simulations = st.slider("Number of simulations", 100, 5000, 1000, 100)
    forecast_days = st.slider("Forecast horizon (days)", 30, 756, 252)
    
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running simulations..."):
            # Use historical mean and covariance
            mean_returns = returns[prices.columns].mean()
            cov_matrix = returns[prices.columns].cov()
            
            # Current weights
            current_weights = weights.iloc[-1].values
            
            # Simulate
            simulated_paths = []
            for _ in range(n_simulations):
                path = [1.0]
                for _ in range(forecast_days):
                    asset_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
                    port_return = np.dot(current_weights, asset_returns)
                    path.append(path[-1] * (1 + port_return))
                simulated_paths.append(path)
            
            simulated_paths = np.array(simulated_paths)
            
            # Plot
            fig_mc = go.Figure()
            
            # Plot percentiles
            percentiles = [5, 25, 50, 75, 95]
            colors = ['red', 'orange', 'blue', 'lightgreen', 'green']
            
            for p, color in zip(percentiles, colors):
                percentile_path = np.percentile(simulated_paths, p, axis=0)
                fig_mc.add_trace(go.Scatter(
                    x=list(range(forecast_days + 1)),
                    y=percentile_path,
                    name=f'{p}th percentile',
                    line=dict(color=color)
                ))
            
            fig_mc.update_layout(
                height=500,
                title=f"Monte Carlo Simulation ({n_simulations} paths, {forecast_days} days)",
                xaxis_title="Days",
                yaxis_title="Portfolio Value (Normalized)"
            )
            st.plotly_chart(fig_mc, width='stretch')
            
            # Statistics
            final_values = simulated_paths[:, -1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Median Final Value", f"{np.median(final_values):.2f}x")
            col2.metric("5th Percentile", f"{np.percentile(final_values, 5):.2f}x")
            col3.metric("95th Percentile", f"{np.percentile(final_values, 95):.2f}x")
            col4.metric("Probability of Loss", f"{(final_values < 1).sum() / n_simulations:.1%}")

with tab_adv3:
    st.subheader("Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download Performance Report (CSV)"):
            report_data = {
                'Date': cum_returns.index,
                'Cumulative_Return': cum_returns.values,
                'Daily_Return': port_returns.values,
                'Drawdown': drawdown.values
            }
            for col in weights.columns:
                report_data[f'Weight_{col}'] = weights[col].values
            
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"portfolio_report_{strategy_choice}_{date.today()}.csv",
                mime='text/csv'
            )
    
    with col2:
        if st.button("📈 Download Metrics Summary (JSON)"):
            summary = {
                "strategy": strategy_choice,
                "weighting": weight_method,
                "period": f"{start_date} to {end_date}",
                "assets": selected_labels,
                "metrics": {
                    "cagr": float(CAGR(port_returns.dropna())),
                    "volatility": float(annualized_vol(port_returns.dropna())),
                    "sharpe": float(sharpe_ratio(port_returns.dropna(), risk_free_rate)),
                    "sortino": float(sortino_ratio(port_returns.dropna(), risk_free_rate)),
                    "max_drawdown": float(max_drawdown(cum_returns)),
                    "calmar": float(calmar_ratio(port_returns.dropna(), cum_returns)),
                    "var_95": float(historical_var(port_returns.dropna(), 0.95)),
                    "cvar_95": float(conditional_var(port_returns.dropna(), 0.95))
                }
            }
            
            import json
            json_str = json.dumps(summary, indent=2)
            st.download_button(
                "Download JSON",
                data=json_str,
                file_name=f"portfolio_metrics_{date.today()}.json",
                mime='application/json'
            )
    
    # Quick summary
    st.subheader("Portfolio Summary")
    summary_text = f"""
    **Strategy:** {strategy_choice}  
    **Weighting:** {weight_method}  
    **Rebalancing:** {rebalance_freq}  
    **Period:** {start_date} to {end_date}  
    **Assets:** {len(selected_tickers)}  
    
    **Performance Metrics:**
    - CAGR: {CAGR(port_returns.dropna()):.2%}
    - Sharpe Ratio: {sharpe_ratio(port_returns.dropna(), risk_free_rate):.2f}
    - Max Drawdown: {max_drawdown(cum_returns):.2%}
    - Final Value: ₹{initial_capital * cum_returns.iloc[-1]:,.0f}
    
    **Risk Metrics:**
    - Volatility: {annualized_vol(port_returns.dropna()):.2%}
    - VaR (95%): {historical_var(port_returns.dropna(), 0.95):.2%}
    - CVaR (95%): {conditional_var(port_returns.dropna(), 0.95):.2%}
    """
    st.markdown(summary_text)

# Footer
st.markdown("---")
st.markdown("""
### 📚 About This Tool
This advanced backtesting platform provides institutional-grade portfolio analysis for Indian markets:
- **7 Trading Strategies**: From simple buy & hold to advanced mean reversion
- **5 Optimization Methods**: Equal weight, inverse vol, risk parity, max Sharpe, min variance  
- **Comprehensive Risk Analytics**: VaR, CVaR, Sortino, Calmar, Omega ratios
- **Advanced Features**: Monte Carlo simulation, stress testing, correlation analysis
- **Real Market Data**: Via Yahoo Finance (NSE tickers use .NS suffix)

**Note:** Past performance does not guarantee future results. This tool is for educational purposes only.
""")

st.caption("Built with Streamlit • Data from Yahoo Finance • Made for Indian Markets 🇮🇳")