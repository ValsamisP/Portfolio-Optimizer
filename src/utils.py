"""
Utility functions for portfolio optimization including mathematical helpers,
data validation, format conversion, and common calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import re
import warnings
import logging

logger = logging.getLogger(__name__)

# Mathematical Utilities
def ensure_symmetric(matrix: np.ndarray) -> np.ndarray:
    """
    Ensure matrix is symmetric by averaging with its transpose
    
    Args:
        matrix: Input matrix
        
    Returns:
        Symmetric matrix
    """
    return (matrix + matrix.T) / 2

def is_positive_definite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if matrix is positive definite
    
    Args:
        matrix: Input matrix
        tol: Tolerance for eigenvalue check
        
    Returns:
        True if positive definite
    """
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues > tol)
    except np.linalg.LinAlgError:
        return False

def condition_number(matrix: np.ndarray) -> float:
    """
    Calculate condition number of matrix
    
    Args:
        matrix: Input matrix
        
    Returns:
        Condition number
    """
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float('inf')

def nearest_correlation_matrix(correlation_matrix: np.ndarray, 
                             max_iterations: int = 100,
                             tolerance: float = 1e-6) -> np.ndarray:
    """
    Find nearest correlation matrix using Higham's algorithm
    
    Args:
        correlation_matrix: Input correlation matrix
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Nearest valid correlation matrix
    """
    n = correlation_matrix.shape[0]
    
    # Initialize
    Y = correlation_matrix.copy()
    
    for i in range(max_iterations):
        # Project onto positive semidefinite matrices
        eigenvals, eigenvecs = np.linalg.eigh(Y)
        eigenvals = np.maximum(eigenvals, 0)
        
        X = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Project onto correlation matrices (diagonal = 1)
        np.fill_diagonal(X, 1.0)
        
        # Check convergence
        if np.linalg.norm(X - Y, 'fro') < tolerance:
            break
            
        Y = X
    
    return X

# Data Validation Utilities
def validate_weights(weights: Union[pd.Series, np.ndarray], 
                    tolerance: float = 1e-6) -> Dict[str, bool]:
    """
    Validate portfolio weights
    
    Args:
        weights: Portfolio weights
        tolerance: Numerical tolerance
        
    Returns:
        Dictionary with validation results
    """
    if isinstance(weights, pd.Series):
        weights_array = weights.values
    else:
        weights_array = weights
    
    validation = {
        'sum_to_one': abs(weights_array.sum() - 1.0) < tolerance,
        'all_non_negative': np.all(weights_array >= -tolerance),
        'no_nan_values': not np.any(np.isnan(weights_array)),
        'no_inf_values': not np.any(np.isinf(weights_array)),
        'within_bounds': np.all((weights_array >= -tolerance) & (weights_array <= 1 + tolerance))
    }
    
    validation['all_valid'] = all(validation.values())
    
    return validation

def validate_returns_covariance(returns: pd.Series, covariance: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate consistency between returns and covariance matrix
    
    Args:
        returns: Expected returns
        covariance: Covariance matrix
        
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # Dimension check
    validation['matching_dimensions'] = len(returns) == len(covariance)
    
    # Index alignment
    validation['matching_indices'] = (
        returns.index.equals(covariance.index) and 
        returns.index.equals(covariance.columns)
    )
    
    # Covariance matrix properties
    validation['covariance_symmetric'] = np.allclose(covariance.values, covariance.values.T)
    validation['covariance_positive_definite'] = is_positive_definite(covariance.values)
    
    # No missing values
    validation['no_nan_returns'] = not returns.isnull().any()
    validation['no_nan_covariance'] = not covariance.isnull().any().any()
    
    validation['all_valid'] = all(validation.values())
    
    return validation

def clean_ticker_symbol(ticker: str) -> str:
    """
    Clean and standardize ticker symbol
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        Cleaned ticker symbol
    """
    if not isinstance(ticker, str):
        return str(ticker).upper().strip()
    
    # Remove whitespace and convert to uppercase
    cleaned = ticker.strip().upper()
    
    # Remove common suffixes that might cause issues
    cleaned = re.sub(r'\.(TO|L|PA|MI|AS|SW|HK)$', '', cleaned)
    
    # Remove special characters except . and -
    cleaned = re.sub(r'[^A-Z0-9.-]', '', cleaned)
    
    return cleaned

# Format Conversion Utilities
def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string
    
    Args:
        value: Decimal value (0.05 = 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, currency: str = "$", decimals: int = 2) -> str:
    """
    Format value as currency string
    
    Args:
        value: Numerical value
        currency: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if abs(value) >= 1e6:
        return f"{currency}{value/1e6:.{max(0, decimals-1)}f}M"
    elif abs(value) >= 1e3:
        return f"{currency}{value/1e3:.{max(0, decimals)}f}K"
    else:
        return f"{currency}{value:.{decimals}f}"

def format_basis_points(value: float, decimals: int = 1) -> str:
    """
    Format value as basis points
    
    Args:
        value: Decimal value (0.0005 = 5 bps)
        decimals: Number of decimal places
        
    Returns:
        Formatted basis points string
    """
    return f"{value * 10000:.{decimals}f} bps"

# Statistical Utilities
def rolling_correlation(series1: pd.Series, series2: pd.Series, 
                       window: int = 252) -> pd.Series:
    """
    Calculate rolling correlation between two series
    
    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size
        
    Returns:
        Rolling correlation series
    """
    return series1.rolling(window=window).corr(series2)

def expanding_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                          freq: int = 252) -> pd.Series:
    """
    Calculate expanding Sharpe ratio
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        freq: Frequency for annualization
        
    Returns:
        Expanding Sharpe ratio series
    """
    rf_period = risk_free_rate / freq
    excess_returns = returns - rf_period
    
    expanding_mean = excess_returns.expanding().mean() * freq
    expanding_std = returns.expanding().std() * np.sqrt(freq)
    
    return expanding_mean / expanding_std

def calculate_diversification_ratio(weights: pd.Series, 
                                  covariance: pd.DataFrame) -> float:
    """
    Calculate diversification ratio (weighted average volatility / portfolio volatility)
    
    Args:
        weights: Portfolio weights
        covariance: Covariance matrix
        
    Returns:
        Diversification ratio
    """
    # Individual volatilities
    individual_vols = np.sqrt(np.diag(covariance))
    
    # Weighted average volatility
    weighted_avg_vol = np.sum(weights * individual_vols)
    
    # Portfolio volatility
    portfolio_vol = np.sqrt(weights @ covariance @ weights)
    
    if portfolio_vol == 0:
        return 1.0
    
    return float(weighted_avg_vol / portfolio_vol)

# Data Processing Utilities
def align_data(*dataframes: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Align multiple dataframes by common index and columns
    
    Args:
        *dataframes: Variable number of DataFrames
        
    Returns:
        List of aligned DataFrames
    """
    if len(dataframes) < 2:
        return list(dataframes)
    
    # Find common index
    common_index = dataframes[0].index
    for df in dataframes[1:]:
        common_index = common_index.intersection(df.index)
    
    # Find common columns
    common_columns = dataframes[0].columns
    for df in dataframes[1:]:
        common_columns = common_columns.intersection(df.columns)
    
    # Align all dataframes
    aligned = []
    for df in dataframes:
        aligned_df = df.reindex(index=common_index, columns=common_columns)
        aligned.append(aligned_df)
    
    return aligned

def winsorize_data(data: pd.Series, lower_percentile: float = 0.01,
                   upper_percentile: float = 0.99) -> pd.Series:
    """
    Winsorize data by clipping extreme values
    
    Args:
        data: Input data series
        lower_percentile: Lower clipping percentile
        upper_percentile: Upper clipping percentile
        
    Returns:
        Winsorized data series
    """
    lower_bound = data.quantile(lower_percentile)
    upper_bound = data.quantile(upper_percentile)
    
    return data.clip(lower=lower_bound, upper=upper_bound)

# Performance Utilities
def annualize_return(returns: pd.Series, freq: int = 252) -> float:
    """
    Annualize returns using compounding
    
    Args:
        returns: Return series
        freq: Frequency (252 for daily, 12 for monthly)
        
    Returns:
        Annualized return
    """
    if len(returns) == 0:
        return 0.0
    
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    
    if n_periods == 0:
        return 0.0
    
    return float((1 + total_return) ** (freq / n_periods) - 1)

def annualize_volatility(returns: pd.Series, freq: int = 252) -> float:
    """
    Annualize volatility
    
    Args:
        returns: Return series
        freq: Frequency for scaling
        
    Returns:
        Annualized volatility
    """
    return float(returns.std() * np.sqrt(freq))

def calculate_information_coefficient(predicted_returns: pd.Series,
                                    actual_returns: pd.Series) -> float:
    """
    Calculate information coefficient (correlation between predicted and actual returns)
    
    Args:
        predicted_returns: Predicted returns
        actual_returns: Actual returns
        
    Returns:
        Information coefficient
    """
    aligned_data = pd.DataFrame({
        'predicted': predicted_returns,
        'actual': actual_returns
    }).dropna()
    
    if len(aligned_data) < 2:
        return 0.0
    
    return float(aligned_data['predicted'].corr(aligned_data['actual']))

# Error Handling Utilities
def safe_divide(numerator: Union[float, np.ndarray], 
               denominator: Union[float, np.ndarray],
               fill_value: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safe division with handling for zero denominator
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        fill_value: Value to use when denominator is zero
        
    Returns:
        Division result with safe handling
    """
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else fill_value
    else:
        result = np.divide(numerator, denominator, 
                          out=np.full_like(numerator, fill_value, dtype=float),
                          where=denominator != 0)
        return result

def handle_optimization_failure(weights_backup: pd.Series, 
                              error_message: str) -> Dict[str, Any]:
    """
    Handle optimization failure by providing fallback solution
    
    Args:
        weights_backup: Backup weights to use
        error_message: Error message from failed optimization
        
    Returns:
        Dictionary with fallback solution and error info
    """
    logger.warning(f"Optimization failed: {error_message}")
    logger.info("Using equal-weight fallback solution")
    
    return {
        'weights': weights_backup,
        'optimization_success': False,
        'error_message': error_message,
        'fallback_used': True,
        'method': 'equal_weight_fallback'
    }

# Configuration Utilities
def create_default_bounds(n_assets: int, min_weight: float = 0.0, 
                         max_weight: float = 1.0) -> List[Tuple[float, float]]:
    """
    Create default weight bounds for optimization
    
    Args:
        n_assets: Number of assets
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        
    Returns:
        List of bounds tuples
    """
    return [(min_weight, max_weight)] * n_assets

def create_weight_sum_constraint() -> Dict[str, Any]:
    """
    Create constraint for weights to sum to 1
    
    Returns:
        Constraint dictionary for scipy.optimize
    """
    return {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

def validate_optimization_inputs(mu: pd.Series, Sigma: pd.DataFrame,
                               weights: Optional[pd.Series] = None) -> List[str]:
    """
    Validate inputs for portfolio optimization
    
    Args:
        mu: Expected returns
        Sigma: Covariance matrix
        weights: Optional portfolio weights
        
    Returns:
        List of validation warnings
    """
    warnings_list = []
    
    # Check basic properties
    if mu.empty:
        warnings_list.append("Expected returns series is empty")
    
    if Sigma.empty:
        warnings_list.append("Covariance matrix is empty")
    
    # Check dimensions
    if len(mu) != len(Sigma):
        warnings_list.append("Dimension mismatch between returns and covariance matrix")
    
    # Check for missing values
    if mu.isnull().any():
        warnings_list.append("Missing values in expected returns")
    
    if Sigma.isnull().any().any():
        warnings_list.append("Missing values in covariance matrix")
    
    # Check covariance matrix properties
    if not np.allclose(Sigma.values, Sigma.values.T):
        warnings_list.append("Covariance matrix is not symmetric")
    
    if not is_positive_definite(Sigma.values):
        warnings_list.append("Covariance matrix is not positive definite")
    
    # Check condition number
    cond_num = condition_number(Sigma.values)
    if cond_num > 1e12:
        warnings_list.append(f"Covariance matrix is ill-conditioned (condition number: {cond_num:.2e})")
    
    # Validate weights if provided
    if weights is not None:
        weight_validation = validate_weights(weights)
        if not weight_validation['all_valid']:
            failed_checks = [k for k, v in weight_validation.items() if not v and k != 'all_valid']
            warnings_list.append(f"Weight validation failed: {', '.join(failed_checks)}")
    
    return warnings_list