# config/settings.py
"""
Configuration settings for the Portfolio Optimizer application.
Contains default values, constraints, and application parameters.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import os

@dataclass
class DataConfig:
    """Data-related configuration parameters"""
    default_tickers: List[str] = None
    min_history_days: int = 252  # Minimum days for stable statistics
    max_assets: int = 20  # Maximum number of assets to prevent performance issues
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    
    def __post_init__(self):
        if self.default_tickers is None:
            self.default_tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META",
                "NVDA", "V", "JNJ", "JPM", "XOM"
            ]

@dataclass 
class OptimizationConfig:
    """Portfolio optimization parameters"""
    max_weight: float = 0.4  # Maximum weight per asset (40%)
    min_weight: float = 0.0  # Minimum weight per asset
    default_rf_rate: float = 0.02  # Default risk-free rate (2%)
    optimization_method: str = "SLSQP"  # Scipy optimization method
    max_iterations: int = 1000
    tolerance: float = 1e-8
    
@dataclass
class RiskConfig:
    """Risk management parameters"""
    default_var_confidence: float = 0.95  # 95% VaR confidence level
    max_var_confidence: float = 0.99
    min_var_confidence: float = 0.80
    default_rebalance_band: float = 0.20  # 20% drift band
    min_trade_threshold: float = 25.0  # Minimum trade size ($)
    default_transaction_cost_bps: float = 5.0  # 5 basis points
    
@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation parameters"""
    default_days: int = 252  # 1 year horizon
    min_days: int = 21  # Minimum simulation horizon
    max_days: int = 756  # Maximum simulation horizon (3 years)
    default_paths: int = 10000
    min_paths: int = 1000
    max_paths: int = 50000
    default_seed: int = 42
    psd_epsilon: float = 1e-10  # For positive semi-definite matrix repairs
    
@dataclass
class UIConfig:
    """User interface settings"""
    page_title: str = "Portfolio Optimizer"
    layout: str = "wide"
    lookback_years_range: tuple = (2, 10)
    default_lookback_years: int = 5
    
class Settings:
    """Main settings container"""
    
    def __init__(self):
        self.data = DataConfig()
        self.optimization = OptimizationConfig()
        self.risk = RiskConfig()
        self.monte_carlo = MonteCarloConfig()
        self.ui = UIConfig()
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load settings from environment variables if available"""
        # Example: override risk-free rate from environment
        rf_env = os.getenv('PORTFOLIO_RF_RATE')
        if rf_env:
            try:
                self.optimization.default_rf_rate = float(rf_env)
            except ValueError:
                pass
                
        # Override max assets limit
        max_assets_env = os.getenv('PORTFOLIO_MAX_ASSETS')
        if max_assets_env:
            try:
                self.data.max_assets = int(max_assets_env)
            except ValueError:
                pass
    
    def get_ticker_validation_rules(self) -> Dict[str, Any]:
        """Return ticker validation rules"""
        return {
            'max_length': 10,  # Maximum ticker symbol length
            'allowed_chars': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ.-',
            'max_count': self.data.max_assets
        }
    
    def get_optimization_bounds(self, n_assets: int) -> List[tuple]:
        """Generate optimization bounds for given number of assets"""
        return [(self.optimization.min_weight, self.optimization.max_weight)] * n_assets
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any warnings"""
        warnings = []
        
        if self.optimization.max_weight * len(self.data.default_tickers) < 1.0:
            warnings.append("Max weight constraint may prevent full investment")
            
        if self.risk.default_rebalance_band < 0.05:
            warnings.append("Very tight rebalance band may cause excessive trading")
            
        if self.monte_carlo.default_paths < 5000:
            warnings.append("Low Monte Carlo paths may give unstable results")
            
        return warnings

# Global settings instance
settings = Settings()

# Convenience functions for common access patterns
def get_default_tickers() -> List[str]:
    return settings.data.default_tickers.copy()

def get_optimization_constraints(n_assets: int) -> Dict[str, Any]:
    return {
        'bounds': settings.get_optimization_bounds(n_assets),
        'sum_constraint': {'type': 'eq', 'fun': lambda w: sum(w) - 1.0},
        'method': settings.optimization.optimization_method
    }