# IndQuant Engine 🇮🇳

### An Institutional-Grade Portfolio & Strategy Backtesting Platform for Indian Markets

## IndQuant Engine is a sophisticated, interactive web application built with Streamlit for backtesting and analyzing quantitative trading strategies on Indian equities and indices. It moves beyond simple backtesting to provide a full suite of institutional-grade tools, from advanced portfolio optimization methods like Hierarchical Risk Parity (HRP) to machine-learning-based regime analysis.

## ✨ Key Features

This platform provides a comprehensive toolkit for both retail and professional investors to design, test, and analyze their investment ideas with mathematical rigor.

#### 📈 **Core Backtesting & Performance Analysis**

- **Flexible Time Periods:** Backtest over any custom date range or use presets (1Y, 5Y, 10Y, Max).
- **Dynamic Asset Selection:** Choose from a wide, curated universe of Indian stocks and indices.
- **Benchmark Comparison:** Continuously measure performance against a benchmark of your choice (e.g., NIFTY 50).
- **Interactive Performance Charts:** Visualize portfolio growth (log scale), drawdowns, and rolling Sharpe ratios.
- **Monthly Returns Heatmap:** Quickly identify seasonal patterns and performance consistency.

#### 🎯 **Advanced Trading Strategies**

- **Classic Technical Strategies:**
  - **SMA Crossover:** Simple trend-following.
  - **RSI-Based:** Mean-reversion strategy based on overbought/oversold levels.
  - **MACD & Bollinger Bands:** Classic momentum and volatility indicators.
- **Factor-Based Strategies:**
  - **Dual Momentum:** Combines relative strength with an absolute momentum filter to dynamically switch between assets and cash.
- **Risk Management Overlays:**
  - **Volatility Targeting:** A sophisticated overlay that adjusts daily leverage to maintain a constant, user-defined level of portfolio risk.

#### ⚖️ **Institutional-Grade Portfolio Construction**

- **Multiple Weighting Schemes:**
  - **Standard:** Equal Weight, Inverse Volatility.
  - **Modern Portfolio Theory:** Minimum Variance, Maximum Sharpe Ratio.
  - **Machine Learning:** **Hierarchical Risk Parity (HRP)**, a cutting-edge method to create more robust and diversified portfolios.
- **Custom Rebalancing:** Set rebalancing frequency from daily to annually to match your strategy's turnover.
- **Cost Simulation:** Accurately model transaction costs and slippage to see their impact on returns.

#### 🔬 **Deep Analytics & Insights**

- **Comprehensive Risk Metrics:** Analyze over a dozen metrics, including **Sortino and Calmar ratios**, **VaR/CVaR**, skewness, kurtosis, beta, and alpha.
- **Return Contribution Analysis:** A detailed breakdown with a waterfall chart showing which assets were the primary drivers of performance.
- **Market Regime Analysis:** Uses a **Hidden Markov Model (HMM)** to automatically identify distinct market regimes (e.g., Bull, Bear, Volatile) and shows your strategy's performance in each.
- **Monte Carlo Simulation:** Forecast a "cone" of potential future outcomes based on the strategy's historical risk/return profile.
- **Stress Testing:** See how the portfolio would have performed during major historical crises like the COVID crash and demonetization.

#### 📄 **Professional Reporting**

- **Downloadable Data:** Export raw backtest results (daily returns, weights) to CSV.
- **PDF Report Generation:** Generate a comprehensive, multi-page PDF report with a single click, embedding key charts and tables for sharing or record-keeping.

---

## 🛠️ Technology Stack

- **Core Application:** [Python](https://www.python.org/)
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Data Analysis & Numerics:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Financial Data:** [yfinance](https://pypi.org/project/yfinance/)
- **Scientific & Statistical Computing:** [SciPy](https://scipy.org/), [StatsModels](https://www.statsmodels.org/)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/), [hmmlearn](https://hmmlearn.readthedocs.io/)
- **Plotting:** [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/)
- **PDF Generation:** [FPDF2 (pyfpdf)](https://pyfpdf.github.io/fpdf2/)

---

## 🚀 Getting Started

Follow these instructions to set up and run IndQuant Engine on your local machine.

### Prerequisites

- Python 3.8 or higher
- `pip` and `venv`

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/IndQuant-Engine.git
    cd IndQuant-Engine
    ```

2.  **Create and activate a virtual environment:**
    _On Windows:_

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

    _On macOS/Linux:_

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    The project includes a `requirements.txt` file. Install all libraries with one command:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

Your web browser should automatically open to the application's local URL (usually `http://localhost:8501`).

---

## 📖 How to Use

The application is designed to be intuitive:

1.  **Configure:** Use the sidebar on the left to select your assets, backtest period, strategy, and portfolio construction method.
2.  **Analyze:** The app will automatically run the backtest. The results are organized into tabs on the main screen.
3.  **Explore:** Dive deep into the different tabs to understand every aspect of your portfolio's performance, from risk and return to its behavior in different market conditions.
4.  **Export:** Go to the "Export Report" tab to generate a professional PDF summary of your findings.

---

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.** The results of the backtests are based on historical data and do not guarantee future performance. Financial markets are inherently unpredictable. Do not base any real-world investment decisions solely on the output of this application. Always conduct your own thorough research and consult with a qualified financial advisor.
