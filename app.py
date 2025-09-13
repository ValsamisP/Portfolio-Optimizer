# app.py
"""
Main Streamlit application for Portfolio Optimizer.
Orchestrates all modules to provide a comprehensive portfolio optimization interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Import our custom modules
from config.settings import settings, get_default_tickers, get_optimization_constraints
from src.data_handler import get_clean_data, DataHandler
from src.optimizer import optimize_portfolio, PortfolioOptimizer
from src.risk_metrics import RiskMetrics, calculate_portfolio_var, calculate_portfolio_cvar
from src.monte_carlo import run_portfolio_monte_carlo, create_fan_chart_data, MonteCarloEngine
from src.rebalancer import PortfolioRebalancer
from src.utils import (
    validate_weights, format_percentage, format_currency, 
    validate_optimization_inputs, handle_optimization_failure
)

# Configure Streamlit
st.set_page_config(
    page_title=settings.ui.page_title,
    layout=settings.ui.layout,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'monte_carlo_results' not in st.session_state:
        st.session_state.monte_carlo_results = None

def create_sidebar():
    """Create sidebar with input controls"""
    st.sidebar.header("⚙️ Portfolio Settings")
    
    # Asset Selection
    st.sidebar.subheader("📊 Asset Universe")
    default_tickers_str = ",".join(get_default_tickers())
    tickers_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value=default_tickers_str,
        help="Enter stock tickers separated by commas"
    )
    
    # Data Parameters
    st.sidebar.subheader("📈 Data Parameters")
    lookback_years = st.sidebar.slider(
        "Lookback Period (years)",
        min_value=settings.ui.lookback_years_range[0],
        max_value=settings.ui.lookback_years_range[1],
        value=settings.ui.default_lookback_years
    )
    
    # Optimization Parameters
    st.sidebar.subheader("🎯 Optimization")
    optimization_method = st.sidebar.selectbox(
        "Optimization Method",
        ["max_sharpe", "min_variance", "risk_parity"],
        index=0,
        format_func=lambda x: {
            "max_sharpe": "Maximum Sharpe Ratio",
            "min_variance": "Minimum Variance",
            "risk_parity": "Risk Parity"
        }[x]
    )
    
    risk_free_rate = st.sidebar.number_input(
        "Risk-free Rate (%)",
        min_value=0.0,
        max_value=15.0,
        value=settings.optimization.default_rf_rate * 100,
        step=0.1,
        format="%.2f"
    ) / 100.0
    
    max_weight = st.sidebar.slider(
        "Maximum Weight per Asset (%)",
        min_value=5,
        max_value=100,
        value=int(settings.optimization.max_weight * 100),
        step=5
    ) / 100.0
    
    # Risk Analysis Parameters
    st.sidebar.subheader("⚠️ Risk Analysis")
    var_confidence = st.sidebar.slider(
        "VaR Confidence Level",
        min_value=settings.risk.min_var_confidence,
        max_value=settings.risk.max_var_confidence,
        value=settings.risk.default_var_confidence,
        step=0.01,
        format="%.2f"
    )
    
    # Monte Carlo Parameters
    st.sidebar.subheader("🎲 Monte Carlo Simulation")
    mc_days = st.sidebar.number_input(
        "Simulation Horizon (days)",
        min_value=settings.monte_carlo.min_days,
        max_value=settings.monte_carlo.max_days,
        value=settings.monte_carlo.default_days,
        step=21
    )
    
    mc_paths = st.sidebar.number_input(
        "Number of Paths",
        min_value=settings.monte_carlo.min_paths,
        max_value=settings.monte_carlo.max_paths,
        value=settings.monte_carlo.default_paths,
        step=1000
    )
    
    mc_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=settings.monte_carlo.default_seed
    )
    
    return {
        'tickers_input': tickers_input,
        'lookback_years': lookback_years,
        'optimization_method': optimization_method,
        'risk_free_rate': risk_free_rate,
        'max_weight': max_weight,
        'var_confidence': var_confidence,
        'mc_days': int(mc_days),
        'mc_paths': int(mc_paths),
        'mc_seed': int(mc_seed)
    }

def load_and_process_data(tickers_str: str, years: int):
    """Load and process portfolio data"""
    try:
        # Parse tickers
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        
        if len(tickers) > settings.data.max_assets:
            st.error(f"Too many assets. Maximum allowed: {settings.data.max_assets}")
            return None
        
        if not tickers:
            st.error("Please provide at least one ticker symbol")
            return None
        
        # Load data with progress indicator
        with st.spinner("Loading market data..."):
            prices, returns, statistics = get_clean_data(tickers, years)
        
        # Data quality report
        handler = DataHandler()
        quality_report = handler.get_data_quality_report(prices, returns)
        
        return {
            'prices': prices,
            'returns': returns,
            'statistics': statistics,
            'quality_report': quality_report,
            'tickers': list(prices.columns)
        }
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def run_optimization(data: Dict, params: Dict):
    """Run portfolio optimization"""
    try:
        mu = data['statistics']['mu_annual']
        Sigma = data['statistics']['cov_annual']
        
        # Validate inputs
        validation_warnings = validate_optimization_inputs(mu, Sigma)
        if validation_warnings:
            st.warning("Optimization warnings:")
            for warning in validation_warnings:
                st.write(f"⚠️ {warning}")
        
        # Run optimization
        with st.spinner(f"Optimizing portfolio using {params['optimization_method']}..."):
            results = optimize_portfolio(
                method=params['optimization_method'],
                mu=mu,
                Sigma=Sigma,
                risk_free_rate=params['risk_free_rate'],
                min_weight=0.0,
                max_weight=params['max_weight']
            )
        
        # Validate results
        weight_validation = validate_weights(results['weights'])
        if not weight_validation['all_valid']:
            st.warning("Optimization produced invalid weights. Using equal-weight fallback.")
            equal_weights = pd.Series(1/len(mu), index=mu.index)
            results = handle_optimization_failure(equal_weights, "Invalid optimization weights")
        
        return results
        
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        # Fallback to equal weights
        equal_weights = pd.Series(1/len(data['statistics']['mu_annual']), 
                                index=data['statistics']['mu_annual'].index)
        return handle_optimization_failure(equal_weights, str(e))

def calculate_risk_metrics(data: Dict, weights: pd.Series, confidence: float):
    """Calculate comprehensive risk metrics"""
    try:
        returns = data['returns']
        
        # Initialize risk calculator
        risk_calc = RiskMetrics()
        portfolio_returns = risk_calc.portfolio_returns(returns, weights)
        
        # Calculate metrics
        var_1d = calculate_portfolio_var(returns, weights, confidence)
        cvar_1d = calculate_portfolio_cvar(returns, weights, confidence)
        
        # Generate comprehensive report
        risk_report = risk_calc.comprehensive_risk_report(
            portfolio_returns,
            risk_free_rate=0.02  # Will be adjusted inside the function
        )
        
        return {
            'var_1d': var_1d,
            'cvar_1d': cvar_1d,
            'portfolio_returns': portfolio_returns,
            'comprehensive_report': risk_report
        }
        
    except Exception as e:
        st.error(f"Risk calculation failed: {str(e)}")
        return None

def run_monte_carlo_simulation(data: Dict, weights: pd.Series, params: Dict):
    """Run Monte Carlo simulation"""
    try:
        mu = data['statistics']['mu_annual']
        Sigma = data['statistics']['cov_annual']
        
        with st.spinner("Running Monte Carlo simulation..."):
            mc_results = run_portfolio_monte_carlo(
                mu=mu,
                cov_matrix=Sigma,
                weights=weights,
                n_days=params['mc_days'],
                n_paths=params['mc_paths'],
                random_seed=params['mc_seed']
            )
        
        # Create fan chart data
        fan_chart_data = create_fan_chart_data(mc_results)
        
        return {
            'simulation_results': mc_results,
            'fan_chart_data': fan_chart_data
        }
        
    except Exception as e:
        st.error(f"Monte Carlo simulation failed: {str(e)}")
        return None

def display_optimization_results(results: Dict, risk_metrics: Dict):
    """Display optimization results and key metrics"""
    
    # Header
    st.markdown('<div class="main-header">Portfolio Optimization Results</div>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        expected_return = results.get('expected_return', 0) * 100
        st.metric(
            "Expected Return",
            f"{expected_return:.2f}%",
            help="Annualized expected portfolio return"
        )
    
    with col2:
        volatility = results.get('volatility', 0) * 100
        st.metric(
            "Volatility",
            f"{volatility:.2f}%",
            help="Annualized portfolio volatility (risk)"
        )
    
    with col3:
        sharpe_ratio = results.get('sharpe_ratio', 0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe_ratio:.2f}",
            help="Risk-adjusted return measure"
        )
    
    with col4:
        var_1d = risk_metrics.get('var_1d', 0) * 100
        st.metric(
            "1-Day VaR (95%)",
            f"{var_1d:.2f}%",
            help="Potential daily loss at 95% confidence"
        )
    
    # Portfolio Weights
    st.subheader("📊 Optimal Portfolio Weights")
    
    weights_df = pd.DataFrame({
        'Asset': results['weights'].index,
        'Weight': results['weights'].values,
        'Weight (%)': results['weights'].values * 100
    })
    
    # Filter out zero weights for cleaner display
    significant_weights = weights_df[weights_df['Weight'] > 0.001].copy()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart of weights
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(significant_weights['Asset'], significant_weights['Weight (%)'])
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight (%)')
        ax.set_title('Portfolio Allocation')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars by weight size
        colors = plt.cm.viridis(significant_weights['Weight (%)'] / significant_weights['Weight (%)'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Weights table
        st.dataframe(
            significant_weights[['Asset', 'Weight (%)']].style.format({'Weight (%)': '{:.2f}%'}),
            use_container_width=True,
            hide_index=True
        )
    
    # Optimization Details
    with st.expander("🔍 Optimization Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Optimization Status:**")
            success = results.get('optimization_success', False)
            st.write(f"✅ Success: {success}")
            
            method = results.get('method', 'Unknown')
            st.write(f"📈 Method: {method}")
            
            if 'error' in results:
                st.write(f"⚠️ Error: {results['error']}")
        
        with col2:
            st.write("**Weight Validation:**")
            weight_validation = validate_weights(results['weights'])
            for check, passed in weight_validation.items():
                icon = "✅" if passed else "❌"
                st.write(f"{icon} {check.replace('_', ' ').title()}: {passed}")

def display_risk_contributions(weights: pd.Series, Sigma: pd.DataFrame):
    """Show each asset's share of portfolio risk (volatility contribution)."""
    sigma_p = float(np.sqrt(weights.values @ Sigma.values @ weights.values))
    mrc = Sigma.values @ weights.values                 # marginal risk contributions
    rc = (weights.values * mrc) / (sigma_p + 1e-12)     # absolute contribution to vol
    rc_pct = pd.Series(rc, index=weights.index)
    rc_pct = rc_pct / (rc_pct.sum() + 1e-12)            # normalize to ~100%

    st.subheader("📐 Risk Contribution by Asset")
    st.bar_chart(rc_pct.rename("share_of_risk"))
    st.caption("In risk parity these bars should be roughly equal.")

    # Optional: quick dispersion metric (closer to 0 ⇒ closer to parity)
    st.metric("Risk contribution dispersion (std)", f"{rc_pct.std():.2%}")


def display_risk_analysis(risk_metrics: Dict, confidence: float):
    """Display detailed risk analysis"""
    
    st.subheader("⚠️ Risk Analysis")
    
    # Risk Metrics Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_1d = risk_metrics.get('var_1d', 0) * 100
        st.metric(
            f"1-Day VaR ({confidence*100:.0f}%)",
            f"{var_1d:.2f}%",
            help=f"Potential loss not exceeded {confidence*100:.0f}% of the time"
        )
    
    with col2:
        cvar_1d = risk_metrics.get('cvar_1d', 0) * 100
        st.metric(
            f"1-Day CVaR ({confidence*100:.0f}%)",
            f"{cvar_1d:.2f}%",
            help=f"Expected loss when VaR is exceeded"
        )
    
    with col3:
        report = risk_metrics.get('comprehensive_report', {})
        drawdown_info = report.get('drawdown', {})
        max_dd = drawdown_info.get('max_drawdown', 0) * 100
        st.metric(
            "Maximum Drawdown",
            f"{abs(max_dd):.2f}%",
            help="Largest peak-to-trough decline"
        )
    
    # Risk Distribution Chart
    if 'portfolio_returns' in risk_metrics:
        portfolio_returns = risk_metrics['portfolio_returns']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Return distribution histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(portfolio_returns * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(risk_metrics['var_1d'] * 100, color='red', linestyle='--', 
                      label=f'VaR ({confidence*100:.0f}%)')
            ax.axvline(risk_metrics['cvar_1d'] * 100, color='darkred', linestyle='--', 
                      label=f'CVaR ({confidence*100:.0f}%)')
            ax.set_xlabel('Daily Returns (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Portfolio Return Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, color='green')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.set_title('Portfolio Performance Over Time')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

def display_monte_carlo_results(mc_results: Dict, params: Dict):
    """Display Monte Carlo simulation results"""
    
    st.subheader("🎲 Monte Carlo Simulation")
    
    if mc_results is None:
        st.error("Monte Carlo simulation failed")
        return
    
    fan_chart_data = mc_results['fan_chart_data']
    simulation_results = mc_results['simulation_results']
    
    # Fan Chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot percentile bands
    ax.fill_between(fan_chart_data['day'], fan_chart_data['p5'], fan_chart_data['p95'], 
                    alpha=0.2, color='blue', label='90% Confidence Band')
    ax.fill_between(fan_chart_data['day'], fan_chart_data['p25'], fan_chart_data['p75'], 
                    alpha=0.3, color='blue', label='50% Confidence Band')
    ax.plot(fan_chart_data['day'], fan_chart_data['p50'], color='blue', linewidth=2, label='Median')
    
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Days')
    ax.set_ylabel('Portfolio Value (Multiple of Initial)')
    ax.set_title(f'Monte Carlo Simulation: {params["mc_paths"]:,} Paths over {params["mc_days"]} Days')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Simulation Statistics
    final_values = simulation_results['final_wealth']
    final_returns = final_values - 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        median_return = np.median(final_returns) * 100
        st.metric(
            "Median Return",
            f"{median_return:.1f}%",
            help="50th percentile of final returns"
        )
    
    with col2:
        p5_return = np.percentile(final_returns, 5) * 100
        st.metric(
            "5th Percentile",
            f"{p5_return:.1f}%",
            help="Worst case in 95% of scenarios"
        )
    
    with col3:
        p95_return = np.percentile(final_returns, 95) * 100
        st.metric(
            "95th Percentile",
            f"{p95_return:.1f}%",
            help="Best case in 95% of scenarios"
        )
    
    with col4:
        prob_loss = np.mean(final_returns < 0) * 100
        st.metric(
            "Probability of Loss",
            f"{prob_loss:.1f}%",
            help="Chance of losing money over the horizon"
        )

def create_rebalancing_section(data: Dict, weights: pd.Series):
    """Create portfolio rebalancing interface"""
    
    st.subheader("🔄 Portfolio Rebalancing")
    
    # Initialize rebalancer
    rebalancer = PortfolioRebalancer()
    
    # Current Holdings Input
    st.write("**Current Holdings**")
    
    # Create editable holdings table
    tickers = data['tickers']
    holdings_data = []
    
    for ticker in tickers:
        holdings_data.append({
            'Ticker': ticker,
            'Quantity': 0.0
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    # Holdings editor
    edited_holdings = st.data_editor(
        holdings_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=0.001)
        }
    )
    
    # Get latest prices (use last available prices from data)
    latest_prices = data['prices'].iloc[-1]
    
    # Rebalancing Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        drift_band = st.slider(
            "Drift Band (±%)",
            min_value=5,
            max_value=40,
            value=int(settings.risk.default_rebalance_band * 100),
            step=5
        ) / 100.0
    
    with col2:
        min_trade_size = st.number_input(
            "Minimum Trade Size ($)",
            min_value=0.0,
            max_value=1000.0,
            value=settings.risk.min_trade_threshold,
            step=5.0
        )
    
    # Convert holdings to Series
    current_holdings = pd.Series(
        edited_holdings.set_index('Ticker')['Quantity'],
        dtype=float
    ).reindex(tickers).fillna(0)
    
    # Perform rebalancing analysis
    if st.button("🔍 Analyze Rebalancing Needs", type="primary"):
        
        rebalancer.min_trade_size = min_trade_size
        
        try:
            rebalancing_report = rebalancer.full_rebalancing_analysis(
                current_holdings=current_holdings,
                target_weights=weights,
                prices=latest_prices,
                drift_threshold=0.05,
                band_width=drift_band
            )
            
            # Display results
            summary = rebalancing_report['summary']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", summary['total_trades'])
            
            with col2:
                st.metric("Trade Value", format_currency(summary['total_trade_value']))
            
            with col3:
                st.metric("Est. Costs", format_currency(summary['total_estimated_costs']))
            
            # Orders table
            orders = rebalancing_report['orders']
            if not orders.empty:
                st.subheader("📋 Proposed Orders")
                
                # Format orders for display
                display_orders = orders[['ticker', 'action', 'quantity', 'price', 'value_usd']].copy()
                display_orders.columns = ['Ticker', 'Action', 'Quantity', 'Price ($)', 'Value ($)']
                
                st.dataframe(
                    display_orders.style.format({
                        'Price ($)': '{:.2f}',
                        'Value ($)': '{:.2f}',
                        'Quantity': '{:.3f}'
                    }),
                    use_container_width=True
                )
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_orders = orders.to_csv(index=False)
                    st.download_button(
                        "📥 Download Orders CSV",
                        csv_orders,
                        file_name="rebalancing_orders.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    target_weights_df = pd.DataFrame({
                        'Ticker': weights.index,
                        'Target Weight': weights.values
                    })
                    csv_weights = target_weights_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Target Weights CSV",
                        csv_weights,
                        file_name="target_weights.csv",
                        mime="text/csv"
                    )
            
            else:
                st.success("✅ No rebalancing needed! Portfolio is within target bands.")
        
        except Exception as e:
            st.error(f"Rebalancing analysis failed: {str(e)}")

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    params = create_sidebar()
    
    # Main content area
    st.title("🎯 Portfolio Optimizer")
    st.markdown("*Advanced portfolio optimization with Monte Carlo simulation and rebalancing tools*")
    
    # Load and process data
    if st.button("🚀 Run Portfolio Analysis", type="primary", use_container_width=True):
        
        # Load data
        data = load_and_process_data(params['tickers_input'], params['lookback_years'])
        
        if data is None:
            st.stop()
        
        st.session_state.portfolio_data = data
        
        # Display data quality info
        with st.expander("📊 Data Quality Report"):
            quality_report = data['quality_report']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Assets", quality_report['n_assets'])
                st.metric("Trading Days", quality_report['n_days'])
            
            with col2:
                date_range = quality_report['date_range']
                st.write(f"**Date Range:**  \n{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
                
                if quality_report['missing_prices']:
                    st.write("**Missing Data:**")
                    for asset, missing_count in quality_report['missing_prices'].items():
                        st.write(f"- {asset}: {missing_count} days")
            
            with col3:
                avg_corr = quality_report.get('avg_correlation', 0)
                st.metric("Avg Correlation", f"{avg_corr:.3f}")
                
                high_corr = quality_report.get('high_correlations', [])
                if high_corr:
                    st.write(f"**High Correlations (>0.8):** {len(high_corr)} pairs")
        
        # Run optimization
        optimization_results = run_optimization(data, params)
        st.session_state.optimization_results = optimization_results
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(data, optimization_results['weights'], params['var_confidence'])
        
        if risk_metrics:
            # Display results
            display_optimization_results(optimization_results, risk_metrics)
            display_risk_contributions(optimization_results['weights'],data['statistics']['cov_annual'])
            display_risk_analysis(risk_metrics, params['var_confidence'])
            
            # Run Monte Carlo simulation
            mc_results = run_monte_carlo_simulation(data, optimization_results['weights'], params)
            st.session_state.monte_carlo_results = mc_results
            
            if mc_results:
                display_monte_carlo_results(mc_results, params)
            
            # Rebalancing section
            create_rebalancing_section(data, optimization_results['weights'])
    
    # Show cached results if available
    elif st.session_state.optimization_results is not None:
        data = st.session_state.portfolio_data
        optimization_results = st.session_state.optimization_results
        
        # Recalculate risk metrics with current parameters
        risk_metrics = calculate_risk_metrics(data, optimization_results['weights'], params['var_confidence'])
        
        if risk_metrics:
            display_optimization_results(optimization_results, risk_metrics)
            display_risk_contributions(optimization_results['weights'],data['statistics']['cov_annual'])
            display_risk_analysis(risk_metrics, params['var_confidence'])
            
            if st.session_state.monte_carlo_results:
                display_monte_carlo_results(st.session_state.monte_carlo_results, params)
            
            create_rebalancing_section(data, optimization_results['weights'])
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*This tool is for educational and research purposes only. "
        "Not financial advice. Please consult with a qualified financial advisor.*"
    )

if __name__ == "__main__":
    main()