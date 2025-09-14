# Portfolio Optimizer
A comprehensive Python application for portfolio optimization,risk analysis and rebalancing using portfolio theory. Built with Streamlit for a interactive user interface.

## Primary Goals
- Provide a framework for testing and comparing different optimization strategies.
- Understand and demonstrate modern portfolio theory through interactive visualizations.
- Implement and compare different portfolio optimization techniques.


## Features
### Portfolio Optimization
- **Maximum Sharpe Ratio:** Find the portfolio with the best risk-adjusted returns
- **Minimum Variance:** Find the portfolio with the global minimum variance for investors that wants less risks
- **Risk Parity:** Equal risk contribution from all assets.


### Risk Analysis
- **Value at Risk(VaR):** Historical,parametric and Monte Carlo methods.
- **Conditional VaR(CVaR):** Expected shortfall analysis.
- **Maximum Drawdown:** Peak-to-through analysis


### Monte Carlo Simulations
- **Portfolio Forecasting:** Simulate future portfolio performance
- **Graphs:** Visualize uncertainty bands and confidence intervals


### Usage
Running the application via "streamlit run app.py"

### Basic Workflow
1.Select Assets:Enter stock tickers in the side bar

2.Configure Parameters:Set lookback period,optimization method and risk parameters

3.Run analysis:Click "Run Portfolio Analysis" to execute optimization

4.Review Results:Take a look on weights, risk metrics, and Monte Carlo simulations

5.Rebalancing:Use current holdings to generate rebalancing orders

<img width="1913" height="1075" alt="image" src="https://github.com/user-attachments/assets/1f246b8f-bffe-4eda-b279-2a3334b5f931" />

<img width="1884" height="947" alt="image" src="https://github.com/user-attachments/assets/d0c59408-75aa-4cca-822f-2e5814199cd5" />

<img width="1889" height="685" alt="image" src="https://github.com/user-attachments/assets/61869627-18db-4f68-9e2b-c5322fc45e9a" />

<img width="1874" height="767" alt="image" src="https://github.com/user-attachments/assets/6179b831-2b16-47e8-a39f-1c4774c4f60b" />

<img width="1906" height="1078" alt="image" src="https://github.com/user-attachments/assets/a5d38362-cd2a-45e9-9230-589c60c0a792" />


### Key Algorithms

**Maximum Sharpe Ratio Optimization:**
Maximizes the ratio of excess return to volatility -> max (μ - rf) / σ
subject to: Σw = 1, w ≥ 0

**Risk Parity:**
Equalizes risk contributions across assets:RC_i = w_i * (Σw)_i / σ_p = 1/n ∀i

**Monte Carlo Simulations:**
Simulates portfolio paths using:
- Multivariate normal distribution
- Cholesky decomposition for correlations

**Value at Risk:**
Multiple calculation methods:
- Historical -> Empirical quantile of returns
- Parametric -> Normal distribution assumption
- Monte carlo -> Simulation-based estimation

**Data Sources:**
- Yahoo Finance -> Historical stock price data via yfinance package


### Limitation and Assumptions

- Historical Data -> Returns based on historical data (not predictive)
- Normal Distributions -> Some models assume normal return distributions
- Market Impact -> Basic market impact estimation
- Liquidity -> Assumes sufficient liquidity for all assets


## Author
Developed by **Panagiotis Valsamis**, M.Sc. in Data Science candidate and aspiring Data Scientist.

## Contributor
**Panagiotis Akidis**, Master in Finance & Economics providing valuable theoretical insights in finance.

## Disclaimer
This tool is for educational and research purposes only. It is not financial advice and should not be used as the sole basis for investment decisions.
