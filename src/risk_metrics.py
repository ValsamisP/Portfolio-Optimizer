# src/risk_metrics.py
"""
Risk measurement and analysis functions for portfolio optimization.
Includes VaR, CVaR, drawdown analysis, and other risk metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, Union, List
import warnings
import logging

logger = logging.getLogger(__name__)

class RiskMetrics:
    """Comprehensive risk measurement toolkit"""
    
    def __init__(self):
        self.risk_cache = {}
    
    def portfolio_returns(self, returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """
        Calculate portfolio returns given asset returns and weights
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights Series
            
        Returns:
            Portfolio returns Series
        """
        # Align weights with returns columns
        aligned_weights = weights.reindex(returns.columns).fillna(0)
        return returns @ aligned_weights
    
    def value_at_risk(self, returns: pd.Series, confidence: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Return series
            confidence: Confidence level (0.95 = 95% VaR)
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR value (negative number, representing potential loss)
        """
        if returns.empty:
            return 0.0
        
        returns_clean = returns.dropna()
        
        if method == 'historical':
            # Historical simulation VaR
            return float(-np.quantile(returns_clean, 1 - confidence))
        
        elif method == 'parametric':
            # Parametric VaR assuming normal distribution
            mean = returns_clean.mean()
            std = returns_clean.std()
            z_score = stats.norm.ppf(1 - confidence)
            return float(-(mean + z_score * std))
        
        elif method == 'monte_carlo':
            # Monte Carlo VaR (using historical parameters)
            mean = returns_clean.mean()
            std = returns_clean.std()
            
            # Generate random scenarios
            np.random.seed(42)
            mc_returns = np.random.normal(mean, std, 10000)
            return float(-np.quantile(mc_returns, 1 - confidence))
        
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
    
    def conditional_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Return series
            confidence: Confidence level
            
        Returns:
            CVaR value (average loss beyond VaR threshold)
        """
        if returns.empty:
            return 0.0
        
        returns_clean = returns.dropna()
        var_threshold = -self.value_at_risk(returns_clean, confidence, 'historical')
        
        # Average of losses beyond VaR threshold
        tail_losses = returns_clean[returns_clean <= var_threshold]
        
        if len(tail_losses) == 0:
            return float(var_threshold)
        
        return float(-tail_losses.mean())
    
    def maximum_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with drawdown metrics
        """
        if returns.empty:
            return {'max_drawdown': 0.0, 'duration': 0, 'recovery_time': 0}
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown series
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_dd = float(drawdown.min())
        
        # Find the period of maximum drawdown
        max_dd_date = drawdown.idxmin()
        
        # Duration: from peak to trough
        peak_before_max_dd = running_max.loc[:max_dd_date].idxmax()
        duration_days = (max_dd_date - peak_before_max_dd).days if hasattr((max_dd_date - peak_before_max_dd), 'days') else 0
        
        # Recovery time: from trough back to peak
        recovery_mask = (drawdown.loc[max_dd_date:] >= -0.001)  # Within 0.1% of recovery
        recovery_time = 0
        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()
            recovery_time = (recovery_date - max_dd_date).days if hasattr((recovery_date - max_dd_date), 'days') else 0
        
        return {
            'max_drawdown': max_dd,
            'drawdown_duration_days': duration_days,
            'recovery_time_days': recovery_time,
            'max_drawdown_date': max_dd_date,
            'drawdown_series': drawdown
        }
    
    def downside_deviation(self, returns: pd.Series, 
                          target_return: float = 0.0) -> float:
        """
        Calculate downside deviation (volatility of negative excess returns)
        
        Args:
            returns: Return series
            target_return: Target return threshold
            
        Returns:
            Downside deviation
        """
        if returns.empty:
            return 0.0
        
        excess_returns = returns - target_return
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return 0.0
        
        return float(np.sqrt(np.mean(negative_returns ** 2)))
    
    def sortino_ratio(self, returns: pd.Series, 
                     risk_free_rate: float = 0.02,
                     target_return: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (return per unit of downside risk)
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            target_return: Target return (default: risk-free rate)
            
        Returns:
            Sortino ratio
        """
        if returns.empty:
            return 0.0
        
        if target_return is None:
            # Convert annualized risk-free rate to same frequency as returns
            if len(returns) > 300:  # Assume daily if > 300 observations
                target_return = risk_free_rate / 252
            elif len(returns) > 50:  # Assume monthly
                target_return = risk_free_rate / 12
            else:  # Assume annual
                target_return = risk_free_rate
        
        mean_return = returns.mean()
        downside_dev = self.downside_deviation(returns, target_return)
        
        if downside_dev == 0:
            return float('inf') if mean_return > target_return else 0.0
        
        return float((mean_return - target_return) / downside_dev)
    
    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown)
        
        Args:
            returns: Return series
            
        Returns:
            Calmar ratio
        """
        if returns.empty:
            return 0.0
        
        # Annualize returns
        if len(returns) > 300:  # Daily
            annual_return = (1 + returns.mean()) ** 252 - 1
        elif len(returns) > 50:  # Monthly
            annual_return = (1 + returns.mean()) ** 12 - 1
        else:  # Annual
            annual_return = returns.mean()
        
        max_dd = abs(self.maximum_drawdown(returns)['max_drawdown'])
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return float(annual_return / max_dd)
    
    def skewness_kurtosis(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate skewness and kurtosis of returns
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with skewness and kurtosis
        """
        if returns.empty or len(returns) < 3:
            return {'skewness': 0.0, 'kurtosis': 0.0}
        
        returns_clean = returns.dropna()
        
        return {
            'skewness': float(stats.skew(returns_clean)),
            'kurtosis': float(stats.kurtosis(returns_clean))
        }
    
    def beta_alpha(self, returns: pd.Series, benchmark_returns: pd.Series,
                   risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate beta and alpha relative to a benchmark
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Risk-free rate (same frequency as returns)
            
        Returns:
            Dictionary with beta, alpha, and R-squared
        """
        if returns.empty or benchmark_returns.empty:
            return {'beta': 1.0, 'alpha': 0.0, 'r_squared': 0.0}
        
        # Align series
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 10:  # Need minimum observations
            return {'beta': 1.0, 'alpha': 0.0, 'r_squared': 0.0}
        
        # Convert annual risk-free rate to same frequency
        if len(aligned_data) > 300:  # Daily
            rf_period = risk_free_rate / 252
        elif len(aligned_data) > 50:  # Monthly
            rf_period = risk_free_rate / 12
        else:  # Annual
            rf_period = risk_free_rate
        
        # Excess returns
        portfolio_excess = aligned_data['portfolio'] - rf_period
        benchmark_excess = aligned_data['benchmark'] - rf_period
        
        # Linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                benchmark_excess, portfolio_excess
            )
            
            return {
                'beta': float(slope),
                'alpha': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value)
            }
        except Exception as e:
            logger.warning(f"Error calculating beta/alpha: {e}")
            return {'beta': 1.0, 'alpha': 0.0, 'r_squared': 0.0}
    
    def tracking_error(self, returns: pd.Series, 
                      benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error (volatility of excess returns)
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Tracking error
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0
        
        # Align and calculate excess returns
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        return float(excess_returns.std())
    
    def information_ratio(self, returns: pd.Series, 
                         benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio (excess return / tracking error)
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0
        
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        mean_excess = excess_returns.mean()
        tracking_err = excess_returns.std()
        
        if tracking_err == 0:
            return float('inf') if mean_excess > 0 else 0.0
        
        return float(mean_excess / tracking_err)
    
    def comprehensive_risk_report(self, returns: pd.Series,
                                 benchmark_returns: Optional[pd.Series] = None,
                                 risk_free_rate: float = 0.02,
                                 confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """
        Generate comprehensive risk analysis report
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Risk-free rate
            confidence_levels: VaR confidence levels
            
        Returns:
            Comprehensive risk metrics dictionary
        """
        if returns.empty:
            return {}
        
        report = {}
        
        # Basic statistics
        report['basic_stats'] = {
            'mean_return': float(returns.mean()),
            'volatility': float(returns.std()),
            'sharpe_ratio': float((returns.mean() - risk_free_rate/252) / returns.std()) if returns.std() > 0 else 0.0
        }
        
        # VaR and CVaR for different confidence levels
        report['var_cvar'] = {}
        for conf in confidence_levels:
            report['var_cvar'][f'{int(conf*100)}%'] = {
                'var': self.value_at_risk(returns, conf),
                'cvar': self.conditional_var(returns, conf)
            }
        
        # Drawdown analysis
        report['drawdown'] = self.maximum_drawdown(returns)
        
        # Alternative risk measures
        report['alternative_ratios'] = {
            'sortino_ratio': self.sortino_ratio(returns, risk_free_rate),
            'calmar_ratio': self.calmar_ratio(returns),
            'downside_deviation': self.downside_deviation(returns)
        }
        
        # Distribution properties
        report['distribution'] = self.skewness_kurtosis(returns)
        
        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            report['benchmark_comparison'] = {
                **self.beta_alpha(returns, benchmark_returns, risk_free_rate),
                'tracking_error': self.tracking_error(returns, benchmark_returns),
                'information_ratio': self.information_ratio(returns, benchmark_returns)
            }
        
        return report

# Convenience functions
def calculate_portfolio_var(returns: pd.DataFrame, weights: pd.Series,
                           confidence: float = 0.95, method: str = 'historical') -> float:
    """Calculate portfolio VaR given returns and weights"""
    risk_calc = RiskMetrics()
    portfolio_returns = risk_calc.portfolio_returns(returns, weights)
    return risk_calc.value_at_risk(portfolio_returns, confidence, method)

def calculate_portfolio_cvar(returns: pd.DataFrame, weights: pd.Series,
                            confidence: float = 0.95) -> float:
    """Calculate portfolio CVaR given returns and weights"""
    risk_calc = RiskMetrics()
    portfolio_returns = risk_calc.portfolio_returns(returns, weights)
    return risk_calc.conditional_var(portfolio_returns, confidence)