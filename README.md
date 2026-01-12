# Black-Scholes Pro Dashboard

## 1. Project Title & Overview

**Black-Scholes Pro Dashboard** is a comprehensive, professional-grade quantitative finance tool designed for sophisticated options analytics. Built with Python and Streamlit, this application goes beyond basic pricing calculators by integrating advanced features such as Implied Volatility (IV) surfaces, interactive Greeks heatmaps, Monte Carlo simulations for payoff distribution, and model-based historical backtesting.

It serves as a powerful utility for quantitative analysts, traders, and financial engineers to visualize option pricing dynamics, assess risk sensitivities (Greeks), and simulate trading strategies using robust mathematical models. The dashboard bridges the gap between theoretical pricing models and practical market data analysis.

## 2. Live Application

**[Launch Live App Here](https://share.streamlit.io/your-username/black-scholes-dashboard)** 
*(Note: If running locally, please refer to the Installation section.)*

In the live application, users can:
- Analyze real-time (market-delayed) option chains for any ticker supported by Yahoo Finance.
- Visualize 3D-like Volatility Surfaces to understand the skew and term structure.
- Simulate option pricing paths and Probability of Profit (PoP).
- Backtest systematic short-volatility strategies against historical price data.

## 3. Core Features

- **European Option Pricing**: Real-time calculation of Call and Put prices using the Black-Scholes-Merton formula.
- **Advanced Greeks Analytics**: Computation of first and second-order Greeks: Delta ($\Delta$), Gamma ($\Gamma$), Theta ($\Theta$), Vega ($\nu$), and Rho ($\rho$).
- **Implied Volatility (IV) Surface**: Interactive heatmap visualization of IV across all available strikes and expirations to visualize the volatility smile/skew.
- **Greeks Heatmaps**: 2D visualization of key risk metrics (e.g., Vega) across the entire option chain.
- **Short Call P&L Sensitivity**: Dynamic heatmap showing theoretical P&L for short option positions across varying underlying prices and volatility shifts.
- **Monte Carlo Simulation**: Stochastic simulation of underlying price paths using Geometric Brownian Motion (GBM) to estimate expected option payoff and risk profiles.
- **Model-Based Historical Backtest**: A simulation engine that reconstructs historical option prices (using realized volatility) to backtest systematic monthly option selling strategies.
- **AI-Powered Quant Assistant**: Integrated GPT-4o-mini assistant (optional) to provide context-aware explanations of model outputs and risk metrics.

## 4. Financial Models & Methodology

*   **Black-Scholes-Merton Model**: The core pricing engine assumes log-normal underlying price distribution, constant risk-free rates, and frictionless markets.
*   **Greeks Computation**: Analytical derivations of the Black-Scholes partial differential equation are used for precise sensitivity calculations.
*   **Monte Carlo Simulation**: Utilizes **Geometric Brownian Motion (GBM)** equation: $dS_t = \mu S_t dt + \sigma S_t dW_t$. The simulation generates thousands of random price paths to converge on an expected payoff, useful for validating analytical prices and estimating tail risks.
*   **Implied Volatility Extraction**: Uses numerical root-finding algorithms (Brent’s method) to solve for volatility $\sigma$ such that $BS(S, K, T, r, \sigma) = MarketPrice$.
*   **Backtesting Methodology**: Since historical option chain data is expensive and sparse, this dashboard employs a **model-based approach**. It estimates the entry price of historical options by applying the Black-Scholes model to historical spot prices and *realized* volatility (rolling window). Settlement is based on actual historical underlying returns.

## 5. Dashboard Workflow

1.  **Ticker Entry**: User inputs a stock ticker (e.g., SPY, AAPL) and configures analysis parameters (Lookback window, Risk-free rate assumption).
2.  **Data Ingestion**: The app fetches historical price data and the full current option chain via `yfinance`.
3.  **Computational Layer**:
    *   Calculates historical realized volatility.
    *   Iterates through the option chain to compute IV and Greeks for every contract.
4.  **Visualization Layer**:
    *   Renders the IV Surface and Greeks heatmaps using Plotly.
    *   Generates P&L scenarios for selected strikes.
5.  **Simulation & Strategy**:
    *   Runs Monte Carlo paths to show the probability distribution of expiry prices.
    *   Executes a chronological backtest of a systemic "Sell ATM Call" strategy, plotting cumulative equity curves.

## 6. Installation & Setup

**Prerequisites**: Python 3.8+

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/black-scholes-pro.git
    cd black-scholes-pro
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Setup OpenAI API**
    Create a `.env` file in the root directory and add your API Key to enable the AI assistant:
    ```env
    OPENAI_API_KEY=sk-your-key-here
    ```

## 7. Usage Instructions

**Run the Application Locally:**
```bash
streamlit run app.py
```

**Exploration Flows:**
*   **Sidebar**: Configure the ticker symbol, historical data period, backtest trade duration, and Monte Carlo simulation parameters (paths/steps).
*   **Market Data**: Review the spot price, computed historical volatility, and available expirations.
*   **Volatility Analysis**: Interact with the IV Surface heatmap. Hover over cells to see specific IV values for Strike vs. Expiry.
*   **Strategy Simulation**: Scroll down to the Monte Carlo section to see the "Probability of Profit" for a short position. Check the Historical Backtest section to see how a naive short-volatility strategy would have performed over the last 2 years (model-simulated).

## 8. File Structure

*   `app.py`: The main application entry point containing all Streamlit UI code, financial formulas, and visualization logic.
*   `README.md`: Project documentation and usage guide.
*   `requirements.txt`: List of required Python libraries.
*   `.env`: (Optional) Environment variable file for storing sensitive API keys.

## 9. Key Technologies

*   **Streamlit**: For building the interactive web-based dashboard interface.
*   **yfinance**: For fetching historical market data and real-time option chains.
*   **NumPy / SciPy**: For high-performance vector mathematics, statistical distributions (Norm CDF/PDF), and numerical optimization (Brentq).
*   **Pandas**: For structured data manipulation and time-series analysis.
*   **Plotly**: For interactive, production-grade 3D surfaces, heatmaps, and charts.
*   **OpenAI API (Optional)**: For embedding a Large Language Model (LLM) to assist users with financial concepts.

## 10. Limitations & Assumptions

*   **Model Assumptions**: The Black-Scholes model assumes constant volatility and log-normal returns, which does not fully capture real-world phenomena like fat tails or volatility smile (though the IV surface visualizes this discrepancy).
*   **Data Latency**: Data from `yfinance` is not real-time and may be delayed. Option chains may occasionally be incomplete or have wide bid-ask spreads that distort IV calculations.
*   **Backtest Accuracy**: The historical backtest is **synthetic**. It assumes options could have been sold at exactly the theoretical Black-Scholes price derived from trailing realized volatility. Real-world execution would differ due to the volatility risk premium, spread costs, and liquidity.

## 11. Disclaimer

**NOT FINANCIAL ADVICE**. This project is for educational and research purposes only. The calculations, simulations, and backtest results provided by this dashboard should not be relied upon for actual trading or investment decisions. Trading options involves significant risk and is not suitable for all investors.

## 12. License

This project is licensed under the **MIT License**.

## 13. Acknowledgments

*   **Streamlit** for the amazing rapid application development framework.
*   **yfinance** for providing accessible market data.
*   **Plotly** for the powerful graphing libraries.
*   **OpenAI** for the API powering the intelligent assistant.
