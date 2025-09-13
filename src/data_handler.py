"""
Data acquisition, cleaning, and preprocessing for portfolio optimization.
Handles data fetching from Yahoo Finance, quality checks, and return calculations.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityError(Exception):
    """Custom exception for data quality issues"""
    pass

class DataHandler:
    """Handles all data-related operations for portfolio optimization"""
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_price_data(_self, tickers: List[str], years: int) -> pd.DataFrame:
        """
        Fetch historical price data from Yahoo Finance
        
        Args:
            tickers: List of ticker symbols
            years: Number of years of historical data
            
        Returns:
            DataFrame with adjusted close prices
        """
        try:
            logger.info(f"Fetching data for {len(tickers)} tickers over {years} years")
            
            # Download data
            raw_data = yf.download(
                tickers,
                period=f"{years}y",
                interval="1d",
                auto_adjust=True,
                threads=True,
                progress=False
            )
            
            # Handle single vs multiple tickers
            if len(tickers) == 1:
                if "Close" in raw_data.columns:
                    prices = raw_data["Close"].to_frame()
                    prices.columns = tickers
                else:
                    prices = raw_data.to_frame()
                    prices.columns = tickers
            else:
                prices = raw_data["Close"] if "Close" in raw_data.columns else raw_data
                
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise DataQualityError(f"Failed to fetch data: {e}")
    
    def clean_price_data(self, prices: pd.DataFrame, 
                        min_history_days: int = 252) -> pd.DataFrame:
        """
        Clean and validate price data
        
        Args:
            prices: Raw price data
            min_history_days: Minimum required trading days
            
        Returns:
            Cleaned price DataFrame
        """
        logger.info("Cleaning price data")
        
        # Remove completely empty rows/columns
        prices_clean = prices.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        if prices_clean.empty:
            raise DataQualityError("No valid price data after cleaning")
        
        # Forward fill missing values (assumes missing = previous day's price)
        prices_clean = prices_clean.fillna(method='ffill')
        
        # Remove any remaining NaN rows, in most of the cases in the first rows
        prices_clean = prices_clean.dropna()
        
        # Validate minimum history requirement
        if len(prices_clean) < min_history_days:
            raise DataQualityError(
                f"Insufficient history: {len(prices_clean)} days < {min_history_days} required"
            )
        
        # Check for strange price movements (more than 50% daily change)
        returns = prices_clean.pct_change().fillna(0)
        suspicious = (returns.abs() > 0.5).any(axis=1)
        if suspicious.sum() > 0:
            logger.warning(f"Found {suspicious.sum()} days with >50% price movements")
        
        logger.info(f"Cleaned data: {len(prices_clean)} days, {len(prices_clean.columns)} assets")
        return prices_clean
    
    def calculate_returns(self, prices: pd.DataFrame, 
                         return_type: str = "log") -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            prices: Price DataFrame
            return_type: "log" or "simple" returns
            
        Returns:
            DataFrame of returns
        """
        if return_type == "log":
            returns = np.log(prices / prices.shift(1))
        elif return_type == "simple":
            returns = prices.pct_change()
        else:
            raise ValueError("return_type must be 'log' or 'simple'")
        
        # Remove first NaN row and any remaining NaNs
        returns = returns.dropna()
        
        # Check for infinite values
        if np.isinf(returns.values).any():
            logger.warning("Infinite values found in returns, replacing with 0")
            returns = returns.replace([np.inf, -np.inf], 0)
        
        return returns
    
    def get_monthly_returns(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily returns to monthly frequency
        
        Args:
            daily_returns: Daily return DataFrame
            
        Returns:
            Monthly returns DataFrame
        """
        # For log returns, sum to get monthly returns
        # For simple returns, compound them: (1+r1)*(1+r2)-1
        monthly_returns = daily_returns.resample("M").sum()
        return monthly_returns
    
    def compute_statistics(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Compute key statistics from returns
        
        Args:
            returns: Daily returns DataFrame
            
        Returns:
            Dictionary containing various statistics
        """
        # Monthly returns for mean estimation (more stable)
        monthly_rets = self.get_monthly_returns(returns)
        
        # Annualized expected returns (from monthly data)
        mu_annual = monthly_rets.mean() * 12
        
        # Annualized covariance matrix (from daily data)
        cov_annual = returns.cov() * 252
        
        # Additional statistics
        volatility_annual = returns.std() * np.sqrt(252)
        sharpe_approx = mu_annual / volatility_annual
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Maximum drawdown (approximate)
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'mu_annual': mu_annual,
            'cov_annual': cov_annual,
            'volatility_annual': volatility_annual,
            'sharpe_approx': sharpe_approx,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_drawdown': max_drawdown
        }
    
    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate ticker symbols and return valid/invalid lists
        
        Args:
            tickers: List of ticker symbols to validate
            
        Returns:
            Tuple of (valid_tickers, invalid_tickers)
        """
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tickers:
            ticker_clean = ticker.strip().upper()
            
            # Basic format validation
            if not ticker_clean or len(ticker_clean) > 10:
                invalid_tickers.append(ticker)
                continue
                
            # Check if contains only allowed characters
            allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ.-0123456789')
            if not set(ticker_clean).issubset(allowed_chars):
                invalid_tickers.append(ticker)
                continue
                
            valid_tickers.append(ticker_clean)
        
        return valid_tickers, invalid_tickers
    
    def get_data_quality_report(self, prices: pd.DataFrame, 
                               returns: pd.DataFrame) -> Dict[str, any]:
        """
        Generate a comprehensive data quality report
        
        Args:
            prices: Price DataFrame
            returns: Returns DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {}
        
        # Basic info
        report['n_assets'] = len(prices.columns)
        report['n_days'] = len(prices)
        report['date_range'] = (prices.index[0], prices.index[-1])
        
        # Missing data analysis
        price_missing = prices.isnull().sum()
        report['missing_prices'] = price_missing[price_missing > 0].to_dict()
        
        # Return statistics
        report['return_stats'] = {
            'mean_daily_return': returns.mean().to_dict(),
            'volatility_daily': returns.std().to_dict(),
            'min_return': returns.min().to_dict(),
            'max_return': returns.max().to_dict()
        }
        
        # Correlation analysis
        correlation_matrix = returns.corr()
        
        # Find highly correlated pairs (>0.8)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr
                    ))
        
        report['high_correlations'] = high_corr_pairs
        report['avg_correlation'] = correlation_matrix.values[np.triu_indices_from(
            correlation_matrix.values, k=1)].mean()
        
        return report
    
    def get_latest_prices(self, tickers: List[str]) -> pd.Series:
        """
        Get the most recent prices for rebalancing calculations
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Series with latest prices
        """
        try:
            # Fetch just the last few days to get current price
            data = yf.download(tickers, period="5d", interval="1d", 
                             auto_adjust=True, progress=False)
            
            if len(tickers) == 1:
                if "Close" in data.columns:
                    latest = data["Close"].dropna().iloc[-1]
                    return pd.Series([latest], index=tickers)
                else:
                    return pd.Series([data.iloc[-1, 0]], index=tickers)
            else:
                latest_prices = data["Close"].dropna().iloc[-1]
                return latest_prices.reindex(tickers).fillna(0)
                
        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            # Return zeros if fetch fails
            return pd.Series(0.0, index=tickers)

# Convenience function for the main app
def get_clean_data(tickers: List[str], years: int, 
                   return_type: str = "log") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    One-stop function to get clean price data, returns, and statistics
    
    Args:
        tickers: List of ticker symbols
        years: Years of historical data
        return_type: Type of returns to calculate
        
    Returns:
        Tuple of (prices, returns, statistics)
    """
    handler = DataHandler()
    
    # Validate tickers
    valid_tickers, invalid_tickers = handler.validate_tickers(tickers)
    if invalid_tickers:
        st.warning(f"Invalid tickers removed: {', '.join(invalid_tickers)}")
    
    if not valid_tickers:
        raise DataQualityError("No valid tickers provided")
    
    # Fetch and clean data
    raw_prices = handler.fetch_price_data(valid_tickers, years)
    clean_prices = handler.clean_price_data(raw_prices)
    returns = handler.calculate_returns(clean_prices, return_type)
    
    # Compute statistics
    stats = handler.compute_statistics(returns)
    
    return clean_prices, returns, stats