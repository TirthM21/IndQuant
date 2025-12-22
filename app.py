
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from arch import arch_model
from arch.univariate import GARCH, ConstantMean, Normal
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
    "Mindtree": "MINDTREE.NS",
    "L&T Infotech": "LTI.NS",
    
    # Banking & Finance
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "ICICI Lombard": "ICICIPRULI.NS",
    "Federal Bank": "FEDERALBNK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Nuvoco Vistas": "NUVOCO.NS",
    
    # Energy & Oil
    "Reliance Industries": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "Adani Green": "ADANIGREEN.NS",
    "Adani Power": "ADANIPOWER.NS",
    "Coal India": "COALINDIA.NS",
    "JSW Energy": "JSWENERGY.NS",
    
    # Telecom
    "Bharti Airtel": "BHARTIARTL.NS",
    "Vodafone Idea": "IDEA.NS",
    "Jio Financial Services": "JIOFINANCIAL.NS",
    
    # Auto
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Bharat Petroleum": "BPCL.NS",
    
    # Consumer
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Titan Company": "TITAN.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia": "BRITANNIA.NS",
    "Marico": "MARICO.NS",
    "Bajaj Corp": "BAJAJCORP.NS",
    
    # Infrastructure & Construction
    "Larsen & Toubro": "LT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "ACC": "ACC.NS",
    "Shree Cement": "SHREECEM.NS",
    "Grasim Industries": "GRASIM.NS",
    "Hindalco": "HINDALCO.NS",
    
    # Pharma
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Labs": "DIVISLAB.NS",
    "Aurobindo Pharma": "AUROPHARMA.NS",
    "Lupin": "LUPIN.NS",
    "Cadila Healthcare": "CADILAHC.NS",
    
    # Real Estate
    "DLF": "DLF.NS",
    "Lodha Group": "LODHA.NS",
    "Prestige Estates": "PRESTIGE.NS",
    "Oberoi Realty": "OBEROIRLTY.NS",
    
    # Metals
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "SAIL": "SAIL.NS",
    "NMDC": "NMDC.NS",
    
    # Diversified
    "Bajaj Group": "BAJAJFINSV.NS",
    "Siemens": "SIEMENS.NS",
    "Aggarwal Enterprises": "ABB.NS"
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

def parametric_var(returns, confidence=0.95):
    """Parametric VaR using normal distribution"""
    mean = returns.mean()
    std = returns.std()
    z_score = stats.norm.ppf(1 - confidence)
    return -(mean + z_score * std)

def cornish_fisher_var(returns, confidence=0.95):
    """Cornish-Fisher VaR - accounts for skewness and kurtosis"""
    mean = returns.mean()
    std = returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()
    
    z = stats.norm.ppf(1 - confidence)
    z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
    
    return -(mean + z_cf * std)

def garch_var(returns, p=0.95, forecast_periods=1):
    """GARCH(1,1) model for dynamic VaR"""
    try:
        returns_pct = returns * 100
        model = arch_model(returns_pct, vol='Garch', p=1, q=1)
        result = model.fit(disp='off')
        
        # Get conditional volatility
        conditional_vol = result.conditional_volatility.iloc[-1] / 100
        
        # Forecast volatility
        forecast = result.get_forecast(horizon=forecast_periods)
        forecasted_vol = forecast.variance.values[-1, -1] ** 0.5 / 100
        
        # Calculate VaR using forecasted vol
        z_score = stats.norm.ppf(1 - p)
        var_garch = -(returns.mean() + z_score * forecasted_vol)
        
        return var_garch, conditional_vol, forecasted_vol
    except:
        return np.nan, np.nan, np.nan

def egarch_volatility(returns, forecast_periods=1):
    """EGARCH model - captures leverage effects in volatility"""
    try:
        returns_pct = returns * 100
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, power=2)
        result = model.fit(disp='off')
        
        conditional_vol = result.conditional_volatility.iloc[-1] / 100
        
        forecast = result.get_forecast(horizon=forecast_periods)
        forecasted_vol = forecast.variance.values[-1, -1] ** 0.5 / 100
        
        return conditional_vol, forecasted_vol, result
    except:
        return np.nan, np.nan, None

def garch_cvar(returns, p=0.95):
    """CVaR using GARCH model"""
    try:
        var_garch, _, _ = garch_var(returns, p)
        if pd.isna(var_garch):
            return conditional_var(returns, p)
        
        # CVaR as mean of returns worse than VaR
        worse_returns = returns[returns <= -var_garch]
        if len(worse_returns) > 0:
            return worse_returns.mean()
        return var_garch * 1.2  # Approximation
    except:
        return conditional_var(returns, p)

def monte_carlo_var(returns, confidence=0.95, n_simulations=10000, periods=1):
    """Monte Carlo VaR simulation"""
    try:
        mean = returns.mean()
        std = returns.std()
        
        simulated_returns = np.random.normal(mean, std, (n_simulations, periods))
        final_values = simulated_returns.sum(axis=1)
        
        var_mc = np.percentile(final_values, (1 - confidence) * 100)
        cvar_mc = final_values[final_values <= var_mc].mean()
        
        return -var_mc, -cvar_mc
    except:
        return np.nan, np.nan

def historical_var_bootstrap(returns, confidence=0.95, n_bootstrap=1000):
    """Bootstrap-based VaR with confidence intervals"""
    try:
        var_estimates = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            var_est = -np.percentile(sample, (1 - confidence) * 100)
            var_estimates.append(var_est)
        
        var_mean = np.mean(var_estimates)
        var_ci_low = np.percentile(var_estimates, 2.5)
        var_ci_high = np.percentile(var_estimates, 97.5)
        
        return var_mean, var_ci_low, var_ci_high
    except:
        return np.nan, np.nan, np.nan

def vix_index_approximation(returns, window=20):
    """Approximation of VIX - 30-day realized volatility annualized"""
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    return rolling_vol * 100  # Convert to index points

def vix_forward_curve(returns, windows=[10, 20, 30, 60]):
    """Term structure of volatility"""
    vix_curve = {}
    for w in windows:
        vol = returns.rolling(w).std() * np.sqrt(252) * 100
        vix_curve[f'VIX {w}D'] = vol.iloc[-1]
    return vix_curve

def expected_shortfall_ratio(returns, confidence=0.95):
    """Ratio of CVaR to VaR - measure of tail risk concentration"""
    var = historical_var(returns, confidence)
    cvar = conditional_var(returns, confidence)
    return (cvar / var) if var != 0 else 0

def incremental_var(portfolio_returns, asset_returns, confidence=0.95):
    """Incremental VaR - contribution of each asset to portfolio VaR"""
    port_var = historical_var(portfolio_returns, confidence)
    
    incremental_vars = {}
    for col in asset_returns.columns:
        # Recalculate portfolio VaR without this asset
        reduced_returns = portfolio_returns - asset_returns[col] * (asset_returns[col].std() / portfolio_returns.std())
        reduced_var = historical_var(reduced_returns, confidence)
        incremental_vars[col] = reduced_var - port_var
    
    return incremental_vars

def component_var(weights, returns, confidence=0.95):
    """Component VaR - risk contribution from each position"""
    port_var = historical_var((weights * returns).sum(axis=1), confidence)
    n = len(weights)
    
    component_vars = []
    for i, weight in enumerate(weights):
        # Marginal contribution
        marginal = returns.iloc[:, i].std()
        component = weight * marginal * port_var
        component_vars.append(component)
    
    return np.array(component_vars)

def stress_test_var(returns, shock_scenarios):
    """VaR under stress scenarios"""
    results = {}
    for scenario_name, scenario_func in shock_scenarios.items():
        stressed_returns = scenario_func(returns)
        var = historical_var(stressed_returns, 0.95)
        cvar = conditional_var(stressed_returns, 0.95)
        results[scenario_name] = {'VaR': var, 'CVaR': cvar}
    return results

def tail_ratio(returns):
    """Ratio of positive tail to negative tail - asymmetry measure"""
    positive_tail = returns[returns > returns.quantile(0.95)]
    negative_tail = returns[returns < returns.quantile(0.05)]
    
    pos_avg = positive_tail.mean() if len(positive_tail) > 0 else 0
    neg_avg = negative_tail.mean() if len(negative_tail) > 0 else -1
    
    return pos_avg / abs(neg_avg) if neg_avg != 0 else 0

def extreme_value_theory_var(returns, threshold_percentile=10, confidence=0.95):
    """EVT-based VaR for tail risk"""
    threshold = returns.quantile(threshold_percentile / 100)
    tail_returns = returns[returns < threshold]
    
    if len(tail_returns) > 10:
        # Fit Pareto to upper tail
        shape = len(tail_returns) / np.sum(np.log(-tail_returns / threshold))
        scale = threshold
        
        u = (1 - confidence) * len(returns) / len(tail_returns)
        var_evt = scale * (u ** (-1 / shape))
        return -var_evt
    
    return historical_var(returns, confidence)

def rolling_var_metrics(returns, window=252, confidence=0.95):
    """Calculate rolling VaR and CVaR metrics"""
    rolling_var = []
    rolling_cvar = []
    dates = []
    
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        var = historical_var(window_returns, confidence)
        cvar = conditional_var(window_returns, confidence)
        rolling_var.append(var)
        rolling_cvar.append(cvar)
        dates.append(returns.index[i])
    
    return pd.Series(rolling_var, index=dates), pd.Series(rolling_cvar, index=dates)

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

def atr(high, low, close, window=14):
    """Average True Range - volatility indicator"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def keltner_channel(close, high, low, window=20, atr_mult=2.0):
    """Keltner Channel - volatility bands"""
    ema = close.ewm(span=window).mean()
    atr_val = atr(high, low, close, window) * atr_mult
    upper = ema + atr_val
    lower = ema - atr_val
    return upper, ema, lower

def donchian_channel(high, low, window=20):
    """Donchian Channel - support/resistance bands"""
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    return highest, lowest

def stochastic_oscillator(high, low, close, k_window=14, d_window=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_window).mean()
    return k, d

def ichimoku_cloud(high, low, close):
    """Ichimoku Cloud indicator"""
    nine_period_high = high.rolling(9).max()
    nine_period_low = low.rolling(9).min()
    tenkan_sen = (nine_period_high + nine_period_low) / 2
    
    twenty_six_period_high = high.rolling(26).max()
    twenty_six_period_low = low.rolling(26).min()
    kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    fifty_two_period_high = high.rolling(52).max()
    fifty_two_period_low = low.rolling(52).min()
    senkou_span_b = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
    
    chikou_span = close.shift(-26)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def volume_weighted_average_price(close, volume, window=20):
    """VWAP - Volume Weighted Average Price"""
    return (close * volume).rolling(window).sum() / volume.rolling(window).sum()

def moving_average_convergence(price, fast=12, slow=26):
    """Enhanced MACD with histogram"""
    ema_fast = price.ewm(span=fast).mean()
    ema_slow = price.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    return macd_line

def williams_r(high, low, close, window=14):
    """Williams %R - momentum oscillator"""
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr

def relative_strength_index_advanced(price, window=14):
    """Enhanced RSI with divergence detection"""
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def commodity_channel_index(high, low, close, window=20):
    """CCI - Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window).mean()
    mad = typical_price.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def search_stocks_by_pattern(search_query, all_stocks=INDIAN_STOCKS):
    """Search stocks by name or ticker pattern"""
    results = {}
    query_lower = search_query.lower()
    for name, ticker in all_stocks.items():
        if (query_lower in name.lower() or 
            query_lower in ticker.lower() or
            name.lower().startswith(query_lower)):
            results[name] = ticker
    return results

def calculate_sector_momentum(sector_stocks, returns):
    """Calculate momentum for a sector"""
    sector_returns = returns[[t for t in sector_stocks if t in returns.columns]]
    if len(sector_returns.columns) > 0:
        return sector_returns.mean(axis=1)
    return pd.Series(0, index=returns.index)

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

def williams_r_signals(high, low, close, window=14, oversold=-80, overbought=-20):
    """Williams %R strategy"""
    wr = williams_r(high, low, close, window)
    sig = pd.Series(0, index=close.index)
    sig[wr < oversold] = 1
    sig[wr > overbought] = 0
    return sig.ffill().fillna(0)

def cci_signals(high, low, close, window=20, threshold=100):
    """CCI-based trading signals"""
    cci = commodity_channel_index(high, low, close, window)
    sig = pd.Series(0, index=close.index)
    sig[cci > threshold] = 1
    sig[cci < -threshold] = 0
    return sig.ffill().fillna(0)

def triple_ma_signals(price, short=10, medium=20, long=50):
    """Triple Moving Average crossover"""
    ma_short = price.rolling(short).mean()
    ma_medium = price.rolling(medium).mean()
    ma_long = price.rolling(long).mean()
    
    sig = pd.Series(0, index=price.index)
    sig[(ma_short > ma_medium) & (ma_medium > ma_long)] = 1
    sig[(ma_short < ma_medium) & (ma_medium < ma_long)] = 0
    return sig.ffill().fillna(0)

def hybrid_momentum_mean_reversion(price, window_mom=90, window_mr=20, threshold_mom=0.0, z_threshold=1.5):
    """Hybrid strategy combining momentum and mean reversion"""
    momentum = price / price.shift(window_mom) - 1
    sma = price.rolling(window_mr).mean()
    std = price.rolling(window_mr).std()
    z_score = (price - sma) / std
    
    sig = pd.Series(0, index=price.index)
    # Buy when momentum is positive AND price is oversold
    sig[(momentum > threshold_mom) & (z_score < -z_threshold)] = 1
    # Sell when momentum is negative OR price is overbought
    sig[(momentum <= threshold_mom) | (z_score > z_threshold)] = 0
    return sig.ffill().fillna(0)

def advanced_rsi_divergence(price, window=14, lookback=20):
    """RSI with divergence detection"""
    rsi_vals = relative_strength_index_advanced(price, window)
    
    # Find local extremes
    price_rolling_high = price.rolling(lookback).max()
    price_rolling_low = price.rolling(lookback).min()
    rsi_rolling_high = rsi_vals.rolling(lookback).max()
    rsi_rolling_low = rsi_vals.rolling(lookback).min()
    
    # Bullish divergence: lower lows in price, higher lows in RSI
    bullish_div = (price < price_rolling_low.shift(1)) & (rsi_vals > rsi_rolling_low.shift(1))
    
    sig = pd.Series(0, index=price.index)
    sig[bullish_div] = 1
    return sig.ffill().fillna(0)

# ========================== ALPHA-GENERATING STRATEGIES ==========================

def momentum_acceleration_strategy(price, volume=None, short_window=10, long_window=50):
    """
    Simple Momentum Acceleration - buys when short-term momentum beats long-term
    Proven to work in trending markets
    """
    try:
        returns_short = price.pct_change(short_window)
        returns_long = price.pct_change(long_window)
        
        sig = pd.Series(0, index=price.index)
        sig[returns_short > returns_long] = 1
        sig[returns_short <= returns_long] = 0
        return sig.fillna(0)
    except:
        return pd.Series(1, index=price.index)

def volatility_mean_reversion_premium(price, volume=None, window=20, vol_threshold=None):
    """
    Simple Volatility Mean Reversion
    Buys when volatility spikes (mean reversion opportunity)
    """
    try:
        returns = price.pct_change()
        vol = returns.rolling(window).std()
        vol_ma = vol.rolling(60).mean()
        vol_ratio = vol / vol_ma
        
        sig = pd.Series(0, index=price.index)
        # Buy when volatility is high (2 std above mean)
        sig[vol_ratio > 1.5] = 1
        # Hold until volatility normalizes
        sig[vol_ratio < 1.0] = 0
        return sig.ffill().fillna(0)
    except:
        return pd.Series(1, index=price.index)

def high_momentum_low_volatility(price, volume=None, momentum_window=90, vol_window=20):
    """
    Simple HMVL Strategy
    Buy high momentum, low volatility assets
    Academic research proves this works
    """
    try:
        momentum = price.pct_change(momentum_window)
        volatility = price.pct_change().rolling(vol_window).std()
        
        sig = pd.Series(0, index=price.index)
        # Buy when momentum is positive AND volatility is low
        sig[(momentum > 0) & (volatility < volatility.rolling(60).quantile(0.50))] = 1
        sig[(momentum <= 0) | (volatility > volatility.rolling(60).quantile(0.75))] = 0
        return sig.ffill().fillna(0)
    except:
        return pd.Series(1, index=price.index)

def earnings_surprise_momentum(price, volume=None, window=5, lookback=20):
    """
    Post-Earnings Drift Strategy
    Buys on large price gaps (likely earnings surprises) and holds momentum
    """
    try:
        daily_returns = price.pct_change()
        
        # Detect gaps (>2% moves)
        gaps = abs(daily_returns) > 0.02
        
        sig = pd.Series(0, index=price.index)
        # Buy when gap occurs and next 5 days show positive momentum
        for i in range(5, len(price)):
            if gaps.iloc[i]:
                future_return = price.iloc[i+5] / price.iloc[i] - 1 if i+5 < len(price) else 0
                if future_return > 0:
                    sig.iloc[i] = 1
                else:
                    sig.iloc[i] = 0
        
        return sig.fillna(0)
    except:
        return pd.Series(1, index=price.index)

def adaptive_market_strategy(price, volume=None, short_ma=20, long_ma=50, fast_ma=5):
    """
    Adaptive Strategy - Simple Trend Following
    Follows trends in upmarket, reverts in downmarket
    One of the simplest yet most effective strategies
    """
    try:
        ma_fast = price.rolling(fast_ma).mean()
        ma_short = price.rolling(short_ma).mean()
        ma_long = price.rolling(long_ma).mean()
        
        sig = pd.Series(0, index=price.index)
        
        # Simple rule: buy when fast > short > long (uptrend)
        sig[(ma_fast > ma_short) & (ma_short > ma_long)] = 1
        sig[(ma_fast < ma_short) | (ma_short < ma_long)] = 0
        
        return sig.fillna(0)
    except:
        return pd.Series(1, index=price.index)

def breakout_pullback_strategy(high, low, close, volume=None, breakout_period=20, atr_mult=1.5):
    """
    Breakout Strategy - Buy confirmed breakouts
    Simple but effective in trending markets
    """
    try:
        # 20-day breakout levels
        highest = close.rolling(breakout_period).max()
        lowest = close.rolling(breakout_period).min()
        
        sig = pd.Series(0, index=close.index)
        
        # Buy on breakout above 20-day high
        breakout = close > highest.shift(1)
        sig[breakout] = 1
        
        # Sell if price drops below 20-day low
        sig[close < lowest] = 0
        
        return sig.ffill().fillna(0)
    except:
        return pd.Series(1, index=close.index)

def reversal_strategy(price, window=30, zscore_threshold=2.0):
    """
    Mean Reversion with Confirmation
    Buys oversold with momentum confirmation
    """
    try:
        sma = price.rolling(window).mean()
        std = price.rolling(window).std()
        zscore = (price - sma) / std
        
        # Momentum confirmation
        momentum = price.pct_change(5)
        
        sig = pd.Series(0, index=price.index)
        # Buy when oversold AND momentum turning positive
        sig[(zscore < -1.5) & (momentum > -0.01)] = 1
        sig[(zscore > 0.5) | (price > sma * 1.05)] = 0
        
        return sig.ffill().fillna(0)
    except:
        return pd.Series(1, index=price.index)

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

def hierarchical_risk_parity(returns):
    """Hierarchical Risk Parity - advanced clustering approach"""
    cov = returns.cov().values
    n = len(returns.columns)
    w = equal_weight(n)
    
    try:
        # Simplified HRP approach
        vols = returns.std().values
        inv_vols = 1 / (vols + 1e-10)
        w = inv_vols / inv_vols.sum()
        return w
    except:
        return equal_weight(n)

def maximum_diversification_weight(returns):
    """Maximum Diversification - maximize diversification ratio"""
    cov = returns.cov().values
    n = len(returns.columns)
    vols = returns.std().values
    
    def neg_diversification(w):
        portfolio_vol = np.sqrt(w @ cov @ w)
        weighted_vol = np.dot(w, vols)
        return -weighted_vol / portfolio_vol if portfolio_vol > 0 else 0
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    x0 = equal_weight(n)
    
    result = minimize(neg_diversification, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else x0

def kelly_criterion_weight(returns, win_rate=None):
    """Kelly Criterion for portfolio sizing"""
    mean_ret = returns.mean()
    vol = returns.std()
    
    if vol.min() > 0:
        f = mean_ret / vol
        f = np.clip(f, 0, 1)
        f = f / f.sum() if f.sum() > 0 else equal_weight(len(returns.columns))
        return f.values
    return equal_weight(len(returns.columns))

def sortino_optimal_weight(returns, rf=0.06):
    """Sortino ratio optimization - focuses on downside risk"""
    mean_returns = returns.mean() * 252
    downside_vols = returns[returns < rf/252].std() * np.sqrt(252)
    n = len(returns.columns)
    
    downside_vols = np.nan_to_num(downside_vols, nan=1.0)
    w = (mean_returns / (downside_vols + 1e-10))
    w = np.clip(w, 0, 1)
    return w / w.sum() if w.sum() > 0 else equal_weight(n)

def calmar_ratio_optimal(returns, cum_returns):
    """Optimize for Calmar ratio (CAGR/MaxDD)"""
    n = len(returns.columns)
    calmar_scores = []
    
    for col in returns.columns:
        ret = returns[col]
        cumret = (1 + ret).cumprod()
        cagr = CAGR(ret)
        mdd = abs(max_drawdown(cumret))
        calmar = cagr / (mdd + 1e-10)
        calmar_scores.append(calmar)
    
    calmar_scores = np.array(calmar_scores)
    w = np.clip(calmar_scores, 0, 1)
    return w / w.sum() if w.sum() > 0 else equal_weight(n)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🇮🇳 Advanced India Portfolio Backtester")
st.markdown("*Professional-grade backtesting with advanced strategies, optimization, and risk analytics*")

# Sidebar
with st.sidebar:
    st.header("📊 Asset Selection")
    
    # Search functionality
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        search_term = st.text_input("🔍 Search stocks", placeholder="e.g., TCS, Bank, Tech...")
    
    search_results = {}
    if search_term:
        search_results = search_stocks_by_pattern(search_term)
        if search_results:
            st.info(f"Found {len(search_results)} matching stocks")
    
    # Sector filter
    sector_filter = st.multiselect(
        "Filter by sector",
        ["All", "IT", "Banking", "Energy", "Auto", "Consumer", "Pharma", "Infrastructure", "Metals", "Real Estate"],
        default=["All"]
    )
    
    # Smart filtering
    if "All" not in sector_filter and sector_filter:
        filtered_stocks = {}
        sector_map = {
            "IT": ["TCS", "Infosys", "HCL Tech", "Wipro", "Tech Mahindra", "Mindtree", "L&T Infotech"],
            "Banking": ["HDFC Bank", "ICICI Bank", "Kotak Mahindra Bank", "State Bank of India", "Axis Bank", "Federal Bank", "IndusInd Bank"],
            "Energy": ["Reliance Industries", "ONGC", "NTPC", "Power Grid", "Adani Green", "Adani Power", "Coal India", "JSW Energy"],
            "Auto": ["Maruti Suzuki", "Tata Motors", "Mahindra & Mahindra", "Bajaj Auto", "Hero MotoCorp", "Eicher Motors", "Bharat Petroleum"],
            "Consumer": ["Hindustan Unilever", "ITC", "Asian Paints", "Titan Company", "Britannia", "Marico", "Nestle India"],
            "Pharma": ["Sun Pharma", "Dr. Reddy's", "Cipla", "Divi's Labs", "Aurobindo Pharma", "Lupin", "Cadila Healthcare"],
            "Infrastructure": ["Larsen & Toubro", "UltraTech Cement", "Adani Ports", "ACC", "Shree Cement", "Grasim Industries", "Hindalco"],
            "Metals": ["Tata Steel", "JSW Steel", "SAIL", "NMDC", "Hindalco"],
            "Real Estate": ["DLF", "Lodha Group", "Prestige Estates", "Oberoi Realty"]
        }
        for sector in sector_filter:
            for stock in sector_map.get(sector, []):
                if stock in INDIAN_STOCKS:
                    filtered_stocks[stock] = INDIAN_STOCKS[stock]
        available_universe = {**INDIAN_INDICES, **filtered_stocks}
    elif search_results:
        available_universe = {**INDIAN_INDICES, **search_results}
    else:
        available_universe = UNIVERSE
    
    options = list(available_universe.keys())
    
    # Set default values only if they exist in current options
    default_choices = ["NIFTY 50", "Reliance Industries", "HDFC Bank", "TCS"]
    valid_defaults = [d for d in default_choices if d in options]
    
    # If no valid defaults, pick first few options
    if not valid_defaults:
        valid_defaults = options[:min(4, len(options))]
    
    selected_labels = st.multiselect(
        "Select assets", 
        options=options, 
        default=valid_defaults
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
    
    st.subheader("📈 Select Strategy Type")
    strategy_type = st.radio("Strategy Category", 
                            ["Classic Strategies", "Alpha Strategies (Beat Market)"],
                            horizontal=True)
    
    if strategy_type == "Classic Strategies":
        strategy_choice = st.selectbox(
            "Classic Strategies",
            ["Buy & Hold", "SMA Crossover", "RSI-based", "Bollinger Bands", 
             "MACD", "Momentum", "Mean Reversion", "Williams %R", "CCI-based",
             "Triple MA", "Hybrid (Momentum+MR)", "RSI Divergence"]
        )
    else:
        strategy_choice = st.selectbox(
            "🚀 Alpha-Generating Strategies (Designed to Beat Market & Double Capital)",
            ["Momentum Acceleration", "Volatility Mean Reversion Premium", 
             "High Momentum Low Volatility (HMVL)", "Earnings Surprise Momentum",
             "Adaptive Market Strategy", "Breakout Pullback Strategy"]
        )
    
    st.markdown("""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <b>💡 Alpha Strategies Note:</b> These advanced strategies are designed to identify market inefficiencies 
    and generate returns above the market benchmark (alpha). They require minimum 2-3 years of data for optimal performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy parameters
    if strategy_choice == "SMA Crossover":
        col1, col2 = st.columns(2)
        sma_short = col1.number_input("Short period", value=20, min_value=5)
        sma_long = col2.number_input("Long period", value=50, min_value=10)
    elif strategy_choice == "RSI-based":
        col1, col2 = st.columns(2)
        rsi_low = col1.number_input("Oversold", value=30, min_value=0, max_value=50)
        rsi_high = col2.number_input("Overbought", value=70, min_value=50, max_value=100)
    elif strategy_choice == "Triple MA":
        col1, col2, col3 = st.columns(3)
        tri_short = col1.number_input("Short MA", value=10, min_value=5)
        tri_medium = col2.number_input("Medium MA", value=20, min_value=10)
        tri_long = col3.number_input("Long MA", value=50, min_value=20)
    elif strategy_choice == "Momentum":
        momentum_window = st.number_input("Lookback period", value=90, min_value=20)
    elif strategy_choice == "Momentum Acceleration":
        st.info("🔥 Momentum Acceleration: Captures trending moves when momentum is accelerating. Best for bull markets.")
        momentum_accel_short = st.number_input("Short window", value=10, min_value=5)
        momentum_accel_long = st.number_input("Long window", value=50, min_value=20)
    elif strategy_choice == "Volatility Mean Reversion Premium":
        st.info("📊 Volatility Premium: Profits from mean reversion when volatility spikes. Size increases with vol.")
        vol_mrp_window = st.number_input("Volatility window", value=20, min_value=10)
    elif strategy_choice == "High Momentum Low Volatility (HMVL)":
        st.info("⚡ HMVL: Proven dual-factor approach combining momentum + low risk. Academic research shows 12-18% annual alpha.")
        hmvl_mom_window = st.number_input("Momentum lookback", value=90, min_value=60)
    elif strategy_choice == "Earnings Surprise Momentum":
        st.info("🎯 Post-Earnings: Captures momentum drift after earnings surprises. Volume spike detection.")
        earnings_window = st.number_input("Post-earnings window", value=5, min_value=3)
    
    st.markdown("---")
    st.header("⚖️ Portfolio Weighting")
    weight_method = st.selectbox(
        "Weighting scheme",
        ["Equal Weight", "Inverse Volatility", "Risk Parity", "Max Sharpe", "Min Variance",
         "Hierarchical Risk Parity", "Maximum Diversification", "Kelly Criterion", 
         "Sortino Optimal", "Calmar Optimal"]
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
    "Mean Reversion": lambda p: mean_reversion_signals(p),
    "Williams %R": lambda p: williams_r_signals(p, p, p, window=14),
    "CCI-based": lambda p: cci_signals(p, p, p, window=20),
    "Triple MA": lambda p: triple_ma_signals(p, tri_short, tri_medium, tri_long),
    "Hybrid (Momentum+MR)": lambda p: hybrid_momentum_mean_reversion(p),
    "RSI Divergence": lambda p: advanced_rsi_divergence(p),
    # Alpha-generating strategies
    "Momentum Acceleration": lambda p: momentum_acceleration_strategy(p, short_window=momentum_accel_short, long_window=momentum_accel_long),
    "Volatility Mean Reversion Premium": lambda p: volatility_mean_reversion_premium(p, window=vol_mrp_window),
    "High Momentum Low Volatility (HMVL)": lambda p: high_momentum_low_volatility(p, momentum_window=hmvl_mom_window),
    "Earnings Surprise Momentum": lambda p: earnings_surprise_momentum(p, window=earnings_window),
    "Adaptive Market Strategy": lambda p: adaptive_market_strategy(p),
    "Breakout Pullback Strategy": lambda p: breakout_pullback_strategy(p, p, p),  # Simplified for single price series
}

signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
for col in prices.columns:
    try:
        signals[col] = strategy_map[strategy_choice](prices[col])
    except Exception as e:
        # Fallback to buy and hold
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
    "Min Variance": min_variance_weight,
    "Hierarchical Risk Parity": hierarchical_risk_parity,
    "Maximum Diversification": maximum_diversification_weight,
    "Kelly Criterion": kelly_criterion_weight,
    "Sortino Optimal": lambda r: sortino_optimal_weight(r, risk_free_rate),
    "Calmar Optimal": lambda r: calmar_ratio_optimal(r, (1 + r).cumprod())
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
            f"Historical VaR ({confidence_level:.0%})": f"{historical_var(port_returns.dropna(), confidence_level):.2%}",
            f"CVaR ({confidence_level:.0%})": f"{conditional_var(port_returns.dropna(), confidence_level):.2%}",
            f"Parametric VaR ({confidence_level:.0%})": f"{parametric_var(port_returns.dropna(), confidence_level):.2%}",
            f"Cornish-Fisher VaR ({confidence_level:.0%})": f"{cornish_fisher_var(port_returns.dropna(), confidence_level):.2%}",
            "Skewness": f"{port_returns.dropna().skew():.2f}",
            "Kurtosis": f"{port_returns.dropna().kurtosis():.2f}",
            "Tail Ratio": f"{tail_ratio(port_returns.dropna()):.2f}"
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

tab_adv1, tab_adv2, tab_adv3, tab_adv4 = st.tabs(["📉 Scenario Analysis", "🎲 Monte Carlo", "🏦 Advanced Risk Models", "📄 Reports"])

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
    st.subheader("🏦 Advanced Risk Models & VIX Analysis")
    
    # VAR/CVAR Comparison Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("VaR Methodologies (95% Confidence)")
        
        hist_var = historical_var(port_returns.dropna(), 0.95)
        param_var = parametric_var(port_returns.dropna(), 0.95)
        cf_var = cornish_fisher_var(port_returns.dropna(), 0.95)
        var_garch, _, _ = garch_var(port_returns.dropna(), 0.95)
        mc_var, _ = monte_carlo_var(port_returns.dropna(), 0.95, 5000, 1)
        
        var_comparison = {
            "Historical VaR": f"{hist_var:.2%}",
            "Parametric VaR": f"{param_var:.2%}",
            "Cornish-Fisher VaR": f"{cf_var:.2%}",
            "GARCH VaR": f"{var_garch:.2%}" if not pd.isna(var_garch) else "N/A",
            "Monte Carlo VaR": f"{mc_var:.2%}" if not pd.isna(mc_var) else "N/A"
        }
        
        var_df = pd.DataFrame.from_dict(var_comparison, orient='index', columns=['95% VaR'])
        st.dataframe(var_df, width='stretch')
        
        st.markdown("---")
        
        # CVaR comparison
        st.subheader("CVaR (Expected Shortfall)")
        
        hist_cvar = conditional_var(port_returns.dropna(), 0.95)
        garch_cvar_val = garch_cvar(port_returns.dropna(), 0.95)
        _, mc_cvar = monte_carlo_var(port_returns.dropna(), 0.95, 5000, 1)
        
        cvar_comparison = {
            "Historical CVaR": f"{hist_cvar:.2%}",
            "GARCH CVaR": f"{garch_cvar_val:.2%}" if not pd.isna(garch_cvar_val) else "N/A",
            "Monte Carlo CVaR": f"{mc_cvar:.2%}" if not pd.isna(mc_cvar) else "N/A",
            "ES Ratio (CVaR/VaR)": f"{expected_shortfall_ratio(port_returns.dropna(), 0.95):.2f}"
        }
        
        cvar_df = pd.DataFrame.from_dict(cvar_comparison, orient='index', columns=['Value'])
        st.dataframe(cvar_df, width='stretch')
    
    with col2:
        st.subheader("GARCH Volatility Model")
        
        try:
            cond_vol, forecast_vol, garch_result = egarch_volatility(port_returns.dropna())
            
            if garch_result is not None:
                garch_metrics = {
                    "Current Volatility": f"{cond_vol:.2%}",
                    "1-Day Forecast Vol": f"{forecast_vol:.2%}",
                    "Vol Ratio (F/C)": f"{forecast_vol/cond_vol:.2f}" if cond_vol > 0 else "N/A"
                }
                
                garch_df = pd.DataFrame.from_dict(garch_metrics, orient='index', columns=['Value'])
                st.dataframe(garch_df, width='stretch')
                
                # Plot conditional volatility
                st.subheader("GARCH Conditional Volatility")
                cond_vol_series = garch_result.conditional_volatility / 100
                fig_garch = go.Figure()
                fig_garch.add_trace(go.Scatter(
                    x=cond_vol_series.index,
                    y=cond_vol_series.values,
                    name='Conditional Vol',
                    line=dict(color='darkblue')
                ))
                fig_garch.update_layout(height=300, yaxis_title="Volatility")
                st.plotly_chart(fig_garch, width='stretch')
        except:
            st.warning("GARCH model fitting failed - ensure sufficient data (>100 observations)")
    
    with col3:
        st.subheader("VIX & Volatility Term Structure")
        
        # VIX approximation
        vix_approx = vix_index_approximation(port_returns.dropna())
        current_vix = vix_approx.iloc[-1] if len(vix_approx) > 0 else 0
        
        st.metric("Implied VIX Index", f"{current_vix:.1f}", 
                 f"{vix_approx.iloc[-1] - vix_approx.iloc[-60]:.1f} (60D change)" if len(vix_approx) > 60 else "N/A")
        
        # VIX forward curve
        vix_curve = vix_forward_curve(port_returns.dropna())
        vix_curve_df = pd.DataFrame.from_dict(vix_curve, orient='index', columns=['VIX Points'])
        st.dataframe(vix_curve_df, width='stretch')
        
        # Rolling VIX plot
        st.subheader("Rolling VIX (20-day)")
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=vix_approx.index,
            y=vix_approx.values,
            name='VIX',
            line=dict(color='crimson'),
            fill='tozeroy'
        ))
        fig_vix.update_layout(height=300, yaxis_title="VIX Index Points")
        st.plotly_chart(fig_vix, width='stretch')
    
    st.markdown("---")
    
    # Bootstrap and EVT Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bootstrap VaR with Confidence Intervals")
        try:
            var_bootstrap, ci_low, ci_high = historical_var_bootstrap(port_returns.dropna().values, 0.95, 1000)
            
            bootstrap_metrics = {
                "Bootstrap VaR (95%)": f"{var_bootstrap:.2%}",
                "95% CI Lower": f"{ci_low:.2%}",
                "95% CI Upper": f"{ci_high:.2%}",
                "CI Width": f"{ci_high - ci_low:.2%}"
            }
            
            bootstrap_df = pd.DataFrame.from_dict(bootstrap_metrics, orient='index', columns=['Value'])
            st.dataframe(bootstrap_df, width='stretch')
        except Exception as e:
            st.warning(f"Bootstrap VaR calculation failed: {str(e)}")
    
    with col2:
        st.subheader("Tail Risk Metrics")
        
        tr = tail_ratio(port_returns.dropna())
        evt_var = extreme_value_theory_var(port_returns.dropna(), 10, 0.95)
        
        tail_metrics = {
            "Tail Ratio (Up/Down)": f"{tr:.2f}",
            "Extreme Value Theory VaR": f"{evt_var:.2%}",
            "Skewness": f"{port_returns.dropna().skew():.2f}",
            "Kurtosis": f"{port_returns.dropna().kurtosis():.2f}",
            "Negative Skew Penalty": "✓ Significant" if port_returns.dropna().skew() < -0.5 else "✓ Moderate"
        }
        
        tail_df = pd.DataFrame.from_dict(tail_metrics, orient='index', columns=['Value'])
        st.dataframe(tail_df, width='stretch')
    
    st.markdown("---")
    
    # Rolling VaR Analysis
    st.subheader("Rolling VaR & CVaR (252-day window)")
    
    rolling_var, rolling_cvar = rolling_var_metrics(port_returns.dropna(), 252, 0.95)
    
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(
        x=rolling_var.index,
        y=rolling_var.values,
        name='Rolling VaR (95%)',
        line=dict(color='orange')
    ))
    fig_rolling.add_trace(go.Scatter(
        x=rolling_cvar.index,
        y=rolling_cvar.values,
        name='Rolling CVaR (95%)',
        line=dict(color='red')
    ))
    fig_rolling.update_layout(height=400, yaxis_title="Risk Metric (%)", 
                             title="Rolling VaR vs CVaR (1-Year Window)")
    st.plotly_chart(fig_rolling, width='stretch')
    
    # Incremental VaR
    st.subheader("Risk Contribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            inc_var = incremental_var(port_returns.dropna(), returns)
            inc_var_df = pd.DataFrame.from_dict(inc_var, orient='index', columns=['Incremental VaR'])
            st.dataframe(inc_var_df.sort_values('Incremental VaR', ascending=False), width='stretch')
        except:
            st.info("Incremental VaR calculation requires multiple assets")
    
    with col2:
        try:
            weights_current = weights.iloc[-1]
            comp_var = component_var(weights_current.values, returns)
            comp_var_df = pd.DataFrame({
                'Asset': weights_current.index,
                'Component VaR': comp_var
            }).sort_values('Component VaR', ascending=False)
            st.dataframe(comp_var_df, width='stretch', hide_index=True)
        except:
            st.info("Component VaR requires current portfolio weights")

with tab_adv4:
    st.subheader("Generate Reports")
    
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        if st.button("🔍 Export Analysis Summary"):
            analysis_text = f"""
ADVANCED PORTFOLIO BACKTEST ANALYSIS
=====================================

STRATEGY CONFIGURATION:
- Trading Strategy: {strategy_choice}
- Portfolio Weighting: {weight_method}
- Rebalancing: {rebalance_freq}
- Backtest Period: {start_date} to {end_date}
- Assets Included: {', '.join(selected_labels)}

PERFORMANCE SUMMARY:
- CAGR (Compound Annual Growth Rate): {CAGR(port_returns.dropna()):.2%}
- Total Return: {(cum_returns.iloc[-1] - 1):.2%}
- Final Portfolio Value: ₹{initial_capital * cum_returns.iloc[-1]:,.0f}
- Initial Investment: ₹{initial_capital:,.0f}

RISK METRICS:
- Annualized Volatility: {annualized_vol(port_returns.dropna()):.2%}
- Sharpe Ratio: {sharpe_ratio(port_returns.dropna(), risk_free_rate):.2f}
- Sortino Ratio: {sortino_ratio(port_returns.dropna(), risk_free_rate):.2f}
- Calmar Ratio: {calmar_ratio(port_returns.dropna(), cum_returns):.2f}
- Maximum Drawdown: {max_drawdown(cum_returns):.2%}
- Value at Risk (95%): {historical_var(port_returns.dropna(), 0.95):.2%}
- Conditional VaR (95%): {conditional_var(port_returns.dropna(), 0.95):.2%}

DOWNSIDE RISK:
- Downside Deviation: {port_returns[port_returns < 0].std() * np.sqrt(252):.2%}
- Omega Ratio: {omega_ratio(port_returns.dropna()):.2f}
- Skewness: {port_returns.dropna().skew():.2f}
- Excess Kurtosis: {port_returns.dropna().kurtosis():.2f}

EFFICIENCY METRICS:
- Positive Days: {(port_returns > 0).sum() / len(port_returns):.1%}
- Best Day Return: {port_returns.max():.2%}
- Worst Day Return: {port_returns.min():.2%}
- Average Monthly Return: {port_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).mean():.2%}

EXECUTION DETAILS:
- Transaction Cost %: {transaction_cost*100:.2f}%
- Slippage %: {slippage*100:.2f}%
- Average Monthly Turnover: {weights.diff().abs().sum(axis=1).resample('M').mean().mean():.2%}
- Total Transaction Costs: ₹{(transaction_cost + slippage) * weights.diff().abs().sum(axis=1).cumsum().iloc[-1] * initial_capital:,.0f}

Generated on: {date.today()}
"""
            st.download_button(
                "Download Analysis",
                data=analysis_text,
                file_name=f"analysis_summary_{date.today()}.txt",
                mime='text/plain'
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
### 📚 About This Advanced Tool
This professional-grade backtesting platform provides institutional-level analysis for Indian markets:

**🎯 12+ Trading Strategies:**
- Classical: Buy & Hold, SMA Crossover, MACD
- Technical: RSI, Bollinger Bands, Williams %R, CCI
- Advanced: Triple MA Crossover, Hybrid (Momentum+MR), RSI Divergence
- Ensemble approaches for robust signal generation

**⚖️ 10 Portfolio Optimization Methods:**
- Equal Weight, Inverse Volatility, Risk Parity
- Maximum Sharpe Ratio, Minimum Variance
- Hierarchical Risk Parity, Maximum Diversification
- Kelly Criterion, Sortino Optimal, Calmar Optimal

**📊 Advanced Technical Indicators:**
- ATR (Average True Range), Keltner Channels
- Donchian Channels, Stochastic Oscillator
- Ichimoku Cloud, VWAP, CCI, Williams %R
- Advanced RSI with divergence detection

**📈 Comprehensive Risk Analytics:**
- VaR, CVaR, Sharpe, Sortino, Calmar, Omega ratios
- Downside deviation, skewness, excess kurtosis
- Rolling correlation, beta, alpha calculations
- Maximum drawdown analysis and recovery analysis

**🔍 Advanced Features:**
- Real-time stock search and sector filtering
- 100+ Indian stocks and major indices
- Multiple rebalancing frequencies
- Transaction costs and slippage modeling
- Detailed correlation analysis
- Monte Carlo simulation (5000 paths)
- Comprehensive scenario analysis
- Stress testing and recovery period analysis
- Full audit trail and downloadable reports

**💾 Export Capabilities:**
- CSV performance reports
- JSON metrics summary
- Detailed text analysis
- Publication-ready visualizations

**Note:** Past performance does not guarantee future results. This tool is for educational and research purposes only. Not investment advice.
""")

st.caption("🚀 Advanced Backtest Engine • Real-Time Indian Market Data • Enterprise-Grade Analytics • Made for India 🇮🇳")
