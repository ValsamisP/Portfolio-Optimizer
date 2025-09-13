# src/monte_carlo.py
"""
Monte Carlo simulation engine for portfolio scenario analysis and forecasting.
Supports multiple distribution models and robust covariance matrix handling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import LinAlgError
from typing import Dict, Tuple, Optional, Union, List
import warnings
import logging

logger = logging.getLogger(__name__)

class MonteCarloEngine:
    """Monte Carlo simulation engine for portfolio analysis"""
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        self.simulation_cache = {}
    
    def ensure_positive_definite(self, cov_matrix: np.ndarray, 
                                epsilon: float = 1e-10) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite for Cholesky decomposition
        
        Args:
            cov_matrix: Covariance matrix
            epsilon: Regularization parameter
            
        Returns:
            Positive definite covariance matrix
        """
        try:
            # Test if already positive definite
            np.linalg.cholesky(cov_matrix)
            return cov_matrix
        except LinAlgError:
            logger.warning("Covariance matrix not positive definite, applying eigenvalue repair")
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Clip negative eigenvalues
            eigenvalues_clipped = np.maximum(eigenvalues, epsilon)
            
            # Reconstruct matrix
            cov_repaired = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T
            
            # Add small diagonal regularization as additional safety
            return cov_repaired + epsilon * np.eye(cov_matrix.shape[0])
    
    def cholesky_decomposition(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Robust Cholesky decomposition with fallback
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Lower triangular Cholesky factor
        """
        cov_psd = self.ensure_positive_definite(cov_matrix)
        
        try:
            return np.linalg.cholesky(cov_psd)
        except LinAlgError as e:
            logger.error(f"Cholesky decomposition failed even after repair: {e}")
            # Ultimate fallback: use diagonal matrix
            return np.diag(np.sqrt(np.maximum(np.diag(cov_matrix), 1e-8)))
    
    def simulate_normal_returns(self, mu: np.ndarray, cov_matrix: np.ndarray,
                               n_days: int, n_paths: int, 
                               freq_adjustment: float = 252.0) -> np.ndarray:
        """
        Simulate asset returns using multivariate normal distribution
        
        Args:
            mu: Expected returns (annualized)
            cov_matrix: Covariance matrix (annualized)
            n_days: Number of simulation days
            n_paths: Number of simulation paths
            freq_adjustment: Frequency adjustment (252 for daily from annual)
            
        Returns:
            Array of shape (n_days, n_paths, n_assets) with simulated returns
        """
        n_assets = len(mu)
        
        # Convert to daily parameters
        daily_mu = mu / freq_adjustment
        daily_cov = cov_matrix / freq_adjustment
        
        # Cholesky decomposition for correlated random variables
        L = self.cholesky_decomposition(daily_cov)
        
        # Generate independent standard normal random variables
        Z = self.rng.standard_normal(size=(n_days, n_paths, n_assets))
        
        # Transform to correlated variables: X = mu + L @ Z
        # Broadcasting: (n_days, n_paths, n_assets)
        daily_drift = daily_mu.reshape(1, 1, -1)  # Shape: (1, 1, n_assets)
        correlated_shocks = (Z @ L.T)  # Shape: (n_days, n_paths, n_assets)
        
        # Add drift to get final returns
        simulated_returns = daily_drift + correlated_shocks
        
        return simulated_returns
    
    def simulate_t_distribution_returns(self, mu: np.ndarray, cov_matrix: np.ndarray,
                                       degrees_freedom: float, n_days: int, n_paths: int,
                                       freq_adjustment: float = 252.0) -> np.ndarray:
        """
        Simulate returns using multivariate t-distribution for fat tails
        
        Args:
            mu: Expected returns (annualized)
            cov_matrix: Covariance matrix (annualized)
            degrees_freedom: Degrees of freedom for t-distribution
            n_days: Number of simulation days
            n_paths: Number of simulation paths
            freq_adjustment: Frequency adjustment
            
        Returns:
            Array of simulated returns with fat tails
        """
        n_assets = len(mu)
        
        # Daily parameters
        daily_mu = mu / freq_adjustment
        daily_cov = cov_matrix / freq_adjustment
        
        # Adjust covariance for t-distribution
        if degrees_freedom > 2:
            scale_factor = degrees_freedom / (degrees_freedom - 2)
            daily_cov_t = daily_cov / scale_factor
        else:
            daily_cov_t = daily_cov
        
        L = self.cholesky_decomposition(daily_cov_t)
        
        # Generate t-distributed random variables
        # Method: Z = N(0,1) / sqrt(Chi2(df)/df)
        normal_vars = self.rng.standard_normal(size=(n_days, n_paths, n_assets))
        chi2_vars = self.rng.chisquare(degrees_freedom, size=(n_days, n_paths, 1))
        t_vars = normal_vars / np.sqrt(chi2_vars / degrees_freedom)
        
        # Apply correlation structure
        daily_drift = daily_mu.reshape(1, 1, -1)
        correlated_t_shocks = (t_vars @ L.T)
        
        return daily_drift + correlated_t_shocks
    
    def portfolio_simulation(self, mu: pd.Series, cov_matrix: pd.DataFrame,
                           weights: pd.Series, n_days: int, n_paths: int,
                           distribution: str = 'normal', 
                           degrees_freedom: float = 5.0) -> Dict[str, np.ndarray]:
        """
        Simulate portfolio returns and cumulative performance
        
        Args:
            mu: Expected returns for assets
            cov_matrix: Covariance matrix
            weights: Portfolio weights
            n_days: Simulation horizon
            n_paths: Number of Monte Carlo paths
            distribution: 'normal' or 't-distribution'
            degrees_freedom: For t-distribution
            
        Returns:
            Dictionary with simulation results
        """
        # Align data
        assets = mu.index
        weights_aligned = weights.reindex(assets).fillna(0)
        mu_aligned = mu.reindex(assets)
        cov_aligned = cov_matrix.reindex(index=assets, columns=assets)
        
        # Convert to numpy arrays
        mu_array = mu_aligned.values
        cov_array = cov_aligned.values
        weights_array = weights_aligned.values
        
        # Simulate asset returns
        if distribution == 'normal':
            asset_returns = self.simulate_normal_returns(
                mu_array, cov_array, n_days, n_paths
            )
        elif distribution == 't-distribution':
            asset_returns = self.simulate_t_distribution_returns(
                mu_array, cov_array, degrees_freedom, n_days, n_paths
            )
        else:
            raise ValueError("Distribution must be 'normal' or 't-distribution'")
        
        # Calculate portfolio returns: sum over assets dimension
        # asset_returns shape: (n_days, n_paths, n_assets)
        # weights_array shape: (n_assets,)
        portfolio_returns = np.sum(asset_returns * weights_array[np.newaxis, np.newaxis, :], axis=2)
        
        # Calculate cumulative returns (wealth evolution)
        # Starting from 1.0, compound daily returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=0)
        
        # Add initial value of 1.0 at time 0
        initial_values = np.ones((1, n_paths))
        cumulative_wealth = np.vstack([initial_values, cumulative_returns])
        
        return {
            'daily_returns': portfolio_returns,  # Shape: (n_days, n_paths)
            'cumulative_wealth': cumulative_wealth,  # Shape: (n_days+1, n_paths)
            'asset_returns': asset_returns,  # Shape: (n_days, n_paths, n_assets)
            'final_wealth': cumulative_returns[-1, :],  # Shape: (n_paths,)
            'simulation_params': {
                'n_days': n_days,
                'n_paths': n_paths,
                'distribution': distribution,
                'degrees_freedom': degrees_freedom if distribution == 't-distribution' else None
            }
        }
    
    def calculate_percentiles(self, wealth_paths: np.ndarray,
                            percentiles: List[float] = [5, 25, 50, 75, 95]) -> Dict[str, np.ndarray]:
        """
        Calculate percentile bands from Monte Carlo simulation
        
        Args:
            wealth_paths: Array of wealth evolution paths
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary with percentile arrays
        """
        percentile_results = {}
        
        for p in percentiles:
            percentile_results[f'p{int(p)}'] = np.percentile(wealth_paths, p, axis=1)
        
        return percentile_results
    
    def risk_metrics_from_simulation(self, portfolio_returns: np.ndarray,
                                   confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """
        Calculate risk metrics from Monte Carlo simulation
        
        Args:
            portfolio_returns: Simulated portfolio returns
            confidence_levels: VaR/CVaR confidence levels
            
        Returns:
            Dictionary with risk metrics
        """
        # Flatten all simulated returns
        all_returns = portfolio_returns.flatten()
        
        risk_metrics = {}
        
        # VaR and CVaR for each confidence level
        for conf in confidence_levels:
            var_threshold = np.percentile(all_returns, (1 - conf) * 100)
            cvar = np.mean(all_returns[all_returns <= var_threshold])
            
            risk_metrics[f'var_{int(conf*100)}'] = -var_threshold  # Positive number for loss
            risk_metrics[f'cvar_{int(conf*100)}'] = -cvar
        
        # Additional metrics
        risk_metrics['volatility'] = np.std(all_returns)
        risk_metrics['skewness'] = stats.skew(all_returns)
        risk_metrics['kurtosis'] = stats.kurtosis(all_returns)
        
        return risk_metrics
    
    def scenario_analysis(self, mu: pd.Series, cov_matrix: pd.DataFrame,
                         weights: pd.Series, scenarios: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Run multiple scenario simulations (bull, bear, normal markets)
        
        Args:
            mu: Base expected returns
            cov_matrix: Base covariance matrix
            weights: Portfolio weights
            scenarios: Dictionary defining different market scenarios
            
        Returns:
            Dictionary with results for each scenario
        """
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            logger.info(f"Running scenario: {scenario_name}")
            
            # Adjust parameters for scenario
            scenario_mu = mu * scenario_params.get('return_multiplier', 1.0)
            scenario_cov = cov_matrix * scenario_params.get('volatility_multiplier', 1.0)
            
            # Run simulation
            sim_results = self.portfolio_simulation(
                scenario_mu, scenario_cov, weights,
                n_days=scenario_params.get('n_days', 252),
                n_paths=scenario_params.get('n_paths', 10000),
                distribution=scenario_params.get('distribution', 'normal')
            )
            
            # Calculate summary statistics
            final_returns = sim_results['final_wealth'] - 1  # Convert to returns
            
            summary = {
                'mean_return': np.mean(final_returns),
                'median_return': np.median(final_returns),
                'std_return': np.std(final_returns),
                'percentiles': {
                    '5th': np.percentile(final_returns, 5),
                    '25th': np.percentile(final_returns, 25),
                    '75th': np.percentile(final_returns, 75),
                    '95th': np.percentile(final_returns, 95)
                },
                'probability_of_loss': np.mean(final_returns < 0),
                'max_loss': np.min(final_returns),
                'max_gain': np.max(final_returns)
            }
            
            results[scenario_name] = {
                'simulation_data': sim_results,
                'summary_stats': summary
            }
        
        return results
    
    def stress_test(self, mu: pd.Series, cov_matrix: pd.DataFrame,
                   weights: pd.Series, stress_scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Simple stress testing by shocking key parameters
        
        Args:
            mu: Expected returns
            cov_matrix: Covariance matrix
            weights: Portfolio weights
            stress_scenarios: Dictionary of stress multipliers
            
        Returns:
            Dictionary with stress test results
        """
        base_return = float(weights @ mu)
        base_volatility = float(np.sqrt(weights @ cov_matrix @ weights))
        
        stress_results = {'base_return': base_return, 'base_volatility': base_volatility}
        
        for scenario_name, shock_factor in stress_scenarios.items():
            if 'return' in scenario_name.lower():
                shocked_mu = mu * shock_factor
                stressed_return = float(weights @ shocked_mu)
                stress_results[scenario_name] = stressed_return
            
            elif 'volatility' in scenario_name.lower() or 'vol' in scenario_name.lower():
                shocked_cov = cov_matrix * (shock_factor ** 2)
                stressed_vol = float(np.sqrt(weights @ shocked_cov @ weights))
                stress_results[scenario_name] = stressed_vol
        
        return stress_results

# Convenience functions for common use cases
def run_portfolio_monte_carlo(mu: pd.Series, cov_matrix: pd.DataFrame,
                             weights: pd.Series, n_days: int = 252, 
                             n_paths: int = 10000, 
                             random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Run standard portfolio Monte Carlo simulation
    
    Args:
        mu: Expected returns
        cov_matrix: Covariance matrix
        weights: Portfolio weights
        n_days: Simulation horizon
        n_paths: Number of paths
        random_seed: Random seed for reproducibility
        
    Returns:
        Simulation results dictionary
    """
    engine = MonteCarloEngine(random_seed)
    return engine.portfolio_simulation(mu, cov_matrix, weights, n_days, n_paths)

def create_fan_chart_data(simulation_results: Dict[str, np.ndarray],
                         percentiles: List[float] = [5, 25, 50, 75, 95]) -> pd.DataFrame:
    """
    Create data for fan chart visualization
    
    Args:
        simulation_results: Results from portfolio_simulation
        percentiles: Percentiles for fan chart bands
        
    Returns:
        DataFrame with percentile data over time
    """
    wealth_paths = simulation_results['cumulative_wealth']
    
    data = {}
    for p in percentiles:
        data[f'p{int(p)}'] = np.percentile(wealth_paths, p, axis=1)
    
    # Add time index
    n_days = wealth_paths.shape[0] - 1  # Subtract 1 because we include day 0
    data['day'] = range(n_days + 1)
    
    return pd.DataFrame(data)

def get_default_scenarios() -> Dict[str, Dict]:
    """
    Get default market scenario definitions
    
    Returns:
        Dictionary of predefined scenarios
    """
    return {
        'normal_market': {
            'return_multiplier': 1.0,
            'volatility_multiplier': 1.0,
            'n_days': 252,
            'n_paths': 10000,
            'distribution': 'normal'
        },
        'bull_market': {
            'return_multiplier': 1.5,
            'volatility_multiplier': 0.8,
            'n_days': 252,
            'n_paths': 10000,
            'distribution': 'normal'
        },
        'bear_market': {
            'return_multiplier': -0.5,
            'volatility_multiplier': 1.8,
            'n_days': 252,
            'n_paths': 10000,
            'distribution': 't-distribution'
        },
        'crisis_scenario': {
            'return_multiplier': -1.0,
            'volatility_multiplier': 2.5,
            'n_days': 63,  # Quarter
            'n_paths': 10000,
            'distribution': 't-distribution'
        }
    }