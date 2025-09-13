# src/optimizer.py
"""
Portfolio optimization algorithms including Maximum Sharpe, Minimum Variance,
Risk Parity, and other modern portfolio theory implementations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import LinAlgError
from typing import Dict, Tuple, Optional, Callable, List
import warnings
import logging

logger = logging.getLogger(__name__)

class OptimizationError(Exception):
    """Custom exception for optimization failures"""
    pass

class PortfolioOptimizer:
    """Main portfolio optimization class with multiple strategies"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.optimization_results = {}
        
    def _validate_inputs(self, mu: pd.Series, Sigma: pd.DataFrame) -> None:
        """Validate input data for optimization"""
        if mu.empty or Sigma.empty:
            raise OptimizationError("Expected returns or covariance matrix is empty")
        
        if len(mu) != len(Sigma):
            raise OptimizationError("Dimension mismatch between returns and covariance matrix")
        
        if not mu.index.equals(Sigma.index) or not mu.index.equals(Sigma.columns):
            raise OptimizationError("Index mismatch between returns and covariance matrix")
        
        # Check for NaN or infinite values
        if mu.isnull().any() or np.isinf(mu).any():
            raise OptimizationError("Invalid values in expected returns")
        
        if Sigma.isnull().any().any() or np.isinf(Sigma.values).any():
            raise OptimizationError("Invalid values in covariance matrix")
    
    def _ensure_positive_definite(self, Sigma: np.ndarray, 
                                 regularization: float = 1e-8) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite
        
        Args:
            Sigma: Covariance matrix
            regularization: Regularization parameter
            
        Returns:
            Regularized positive definite matrix
        """
        try:
            # Check if already positive definite
            np.linalg.cholesky(Sigma)
            return Sigma
        except LinAlgError:
            logger.warning("Covariance matrix not positive definite, regularizing")
            
            # Eigenvalue decomposition and clipping
            eigenvals, eigenvecs = np.linalg.eigh(Sigma)
            eigenvals_clipped = np.maximum(eigenvals, regularization)
            
            # Reconstruct matrix
            Sigma_reg = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
            
            # Add small diagonal regularization as backup
            return Sigma_reg + regularization * np.eye(len(Sigma))
    
    def _create_constraints(self, n_assets: int, 
                           additional_constraints: Optional[List[Dict]] = None) -> List[Dict]:
        """Create optimization constraints"""
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        if additional_constraints:
            constraints.extend(additional_constraints)
            
        return constraints
    
    def _create_bounds(self, n_assets: int, min_weight: float = 0.0, 
                      max_weight: float = 1.0) -> List[Tuple[float, float]]:
        """Create weight bounds for optimization"""
        return [(min_weight, max_weight)] * n_assets
    
    def maximize_sharpe_ratio(self, mu: pd.Series, Sigma: pd.DataFrame,
                             min_weight: float = 0.0, max_weight: float = 1.0,
                             additional_constraints: Optional[List[Dict]] = None) -> Dict:
        """
        Find portfolio weights that maximize the Sharpe ratio
        
        Args:
            mu: Expected returns
            Sigma: Covariance matrix
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            additional_constraints: Additional optimization constraints
            
        Returns:
            Dictionary with optimization results
        """
        self._validate_inputs(mu, Sigma)
        n_assets = len(mu)
        
        # Regularize covariance matrix
        Sigma_reg = self._ensure_positive_definite(Sigma.values)
        
        # Initial guess: equal weights
        w0 = np.ones(n_assets) / n_assets
        
        # Objective function: negative Sharpe ratio
        def neg_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, mu.values)
            portfolio_variance = np.dot(weights, np.dot(Sigma_reg, weights))
            portfolio_std = np.sqrt(max(portfolio_variance, 1e-12))
            
            excess_return = portfolio_return - self.risk_free_rate
            return -excess_return / portfolio_std
        
        # Constraints and bounds
        constraints = self._create_constraints(n_assets, additional_constraints)
        bounds = self._create_bounds(n_assets, min_weight, max_weight)
        
        # Optimize
        try:
            result = minimize(
                neg_sharpe_ratio,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                # Fallback to equal weights
                weights = pd.Series(w0, index=mu.index)
            else:
                weights = pd.Series(result.x, index=mu.index)
                weights = weights.clip(lower=0)  # Ensure non-negative
                weights = weights / weights.sum()  # Normalize
                
            # Calculate portfolio metrics
            portfolio_return = float(np.dot(weights.values, mu.values))
            portfolio_variance = float(np.dot(weights.values, np.dot(Sigma_reg, weights.values)))
            portfolio_std = float(np.sqrt(portfolio_variance))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': result.success if 'result' in locals() else False,
                'method': 'max_sharpe'
            }
            
        except Exception as e:
            logger.error(f"Sharpe ratio optimization failed: {e}")
            # Return equal weights as fallback
            weights = pd.Series(w0, index=mu.index)
            portfolio_return = float(np.dot(weights.values, mu.values))
            portfolio_variance = float(np.dot(weights.values, np.dot(Sigma_reg, weights.values)))
            portfolio_std = float(np.sqrt(portfolio_variance))
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_std,
                'optimization_success': False,
                'method': 'max_sharpe',
                'error': str(e)
            }
    
    def minimize_variance(self, Sigma: pd.DataFrame,
                         min_weight: float = 0.0, max_weight: float = 1.0,mu: Optional[pd.Series] = None) -> Dict:
        """
        Find minimum variance portfolio (Global Minimum Variance Portfolio)
        
        Args:
            Sigma: Covariance matrix
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(Sigma)
        Sigma_reg = self._ensure_positive_definite(Sigma.values)
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Objective: portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(Sigma_reg, weights))
        
        constraints = self._create_constraints(n_assets)
        bounds = self._create_bounds(n_assets, min_weight, max_weight)
        
        try:
            result = minimize(
                portfolio_variance,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            weights = pd.Series(result.x, index=Sigma.index) if result.success else pd.Series(w0, index=Sigma.index)
            weights = weights.clip(lower=0) / weights.sum()
            
            portfolio_variance_val = float(np.dot(weights.values, np.dot(Sigma_reg, weights.values)))
            portfolio_std = float(np.sqrt(portfolio_variance_val))
            
            out = {
                'weights': weights,
                'volatility': portfolio_std,
                'variance': portfolio_variance_val,
                'optimization_success': result.success,
                'method': 'min_variance'
            }
            if mu is not None:
                exp_ret = float(weights.values @ mu.values)
                sharpe = ((exp_ret - self.risk_free_rate) / portfolio_std) if portfolio_std > 0 else 0.0
                out['expected_return'] = exp_ret
                out['sharpe_ratio'] = sharpe
            return out
            
        except Exception as e:
            logger.error(f"Minimum variance optimization failed: {e}")
            weights = pd.Series(w0, index=Sigma.index)
            portfolio_variance_val = float(np.dot(weights.values, np.dot(Sigma_reg, weights.values)))
            
            return {
                'weights': weights,
                'volatility': float(np.sqrt(portfolio_variance_val)),
                'variance': portfolio_variance_val,
                'optimization_success': False,
                'method': 'min_variance',
                'error': str(e)
            }
    
    def risk_parity(self, Sigma: pd.DataFrame, 
                   target_risk: Optional[pd.Series] = None,min_weight: float = 0.001, max_weight: float = 0.999,mu: Optional[pd.Series] = None) -> Dict:
        """
        Risk Parity (Equal Risk Contribution) portfolio
        
        Args:
            Sigma: Covariance matrix
            target_risk: Target risk contributions (default: equal)
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(Sigma)
        Sigma_reg = self._ensure_positive_definite(Sigma.values)
        
        # Default: equal risk contribution
        if target_risk is None:
            target_risk = pd.Series(1.0 / n_assets, index=Sigma.index)
        else:
            target_risk = target_risk / target_risk.sum()  # Normalize
            
        # Initial guess: inverse volatility weights
        vol = np.sqrt(np.diag(Sigma_reg))
        w0 = (1 / vol) / np.sum(1 / vol)
        
        def risk_parity_objective(weights):
            """Minimize sum of squared differences from target risk contributions"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(Sigma_reg, weights)))
            
            # Risk contributions: w_i * (Sigma @ w)_i / portfolio_vol
            marginal_contrib = np.dot(Sigma_reg, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            risk_contrib = risk_contrib / np.sum(risk_contrib)  # Normalize
            
            # Objective: sum of squared deviations from target
            return np.sum((risk_contrib - target_risk.values) ** 2)
        
        constraints = self._create_constraints(n_assets)
        bounds = self._create_bounds(n_assets, min_weight, max_weight)  # Avoid exact zeros
        
        try:
            result = minimize(
                risk_parity_objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            weights = pd.Series(result.x, index=Sigma.index) if result.success else pd.Series(w0, index=Sigma.index)
            weights = weights / weights.sum()
            
            # Calculate actual risk contributions
            portfolio_vol = np.sqrt(np.dot(weights.values, np.dot(Sigma_reg, weights.values)))
            marginal_contrib = np.dot(Sigma_reg, weights.values)
            risk_contrib = weights.values * marginal_contrib / portfolio_vol
            risk_contrib = pd.Series(risk_contrib / np.sum(risk_contrib), index=Sigma.index)
            
            out = {
                'weights': weights,
                'volatility': float(portfolio_vol),
                'risk_contributions': risk_contrib,
                'optimization_success': result.success,
                'method': 'risk_parity'
            }
            if mu is not None:
                exp_ret = float(weights.values @ mu.values)
                sharpe = ((exp_ret - self.risk_free_rate) / portfolio_vol) if portfolio_vol > 0 else 0.0
                out['expected_return'] = exp_ret
                out['sharpe_ratio'] = sharpe
            return out
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            weights = pd.Series(w0, index=Sigma.index)
            portfolio_vol = np.sqrt(np.dot(weights.values, np.dot(Sigma_reg, weights.values)))
            
            return {
                'weights': weights,
                'volatility': float(portfolio_vol),
                'optimization_success': False,
                'method': 'risk_parity',
                'error': str(e)
            }
    
    def efficient_frontier(self, mu: pd.Series, Sigma: pd.DataFrame,
                          n_points: int = 20, 
                          min_weight: float = 0.0, 
                          max_weight: float = 1.0) -> Dict:
        """
        Generate efficient frontier points
        
        Args:
            mu: Expected returns
            Sigma: Covariance matrix
            n_points: Number of frontier points to generate
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            Dictionary with frontier data
        """
        self._validate_inputs(mu, Sigma)
        
        # Find min variance and max Sharpe portfolios as endpoints
        min_var_result = self.minimize_variance(Sigma, min_weight, max_weight)
        max_sharpe_result = self.maximize_sharpe_ratio(mu, Sigma, min_weight, max_weight)
        
        min_return = np.dot(min_var_result['weights'].values, mu.values)
        max_return = np.dot(max_sharpe_result['weights'].values, mu.values)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return * 1.2, n_points)
        
        frontier_weights = []
        frontier_returns = []
        frontier_volatilities = []
        
        n_assets = len(mu)
        Sigma_reg = self._ensure_positive_definite(Sigma.values)
        
        for target_return in target_returns:
            w0 = np.ones(n_assets) / n_assets
            
            # Minimize variance subject to target return
            def portfolio_variance(weights):
                return np.dot(weights, np.dot(Sigma_reg, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mu.values) - target_return}
            ]
            
            bounds = self._create_bounds(n_assets, min_weight, max_weight)
            
            try:
                result = minimize(
                    portfolio_variance,
                    w0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    weights = pd.Series(result.x, index=mu.index)
                    weights = weights / weights.sum()
                    
                    actual_return = np.dot(weights.values, mu.values)
                    volatility = np.sqrt(np.dot(weights.values, np.dot(Sigma_reg, weights.values)))
                    
                    frontier_weights.append(weights)
                    frontier_returns.append(actual_return)
                    frontier_volatilities.append(volatility)
                    
            except Exception as e:
                logger.warning(f"Failed to optimize for target return {target_return}: {e}")
                continue
        
        return {
            'returns': np.array(frontier_returns),
            'volatilities': np.array(frontier_volatilities),
            'weights': frontier_weights,
            'sharpe_ratios': (np.array(frontier_returns) - self.risk_free_rate) / np.array(frontier_volatilities)
        }
    
    def black_litterman(self, mu_market: pd.Series, Sigma: pd.DataFrame,
                       P: np.ndarray, Q: np.ndarray, 
                       tau: float = 0.025, omega: Optional[np.ndarray] = None) -> Dict:
        """
        Black-Litterman model implementation
        
        Args:
            mu_market: Market equilibrium returns
            Sigma: Covariance matrix
            P: Picking matrix (views)
            Q: View returns
            tau: Uncertainty parameter
            omega: Uncertainty matrix for views
            
        Returns:
            Dictionary with Black-Litterman adjusted parameters
        """
        n_assets = len(mu_market)
        
        if omega is None:
            # Default: diagonal matrix with view uncertainties
            omega = np.eye(len(Q)) * 0.01
        
        # Black-Litterman formula
        Sigma_inv = np.linalg.inv(Sigma.values)
        tau_Sigma = tau * Sigma.values
        
        # Posterior covariance
        M1 = np.linalg.inv(tau_Sigma)
        M2 = P.T @ np.linalg.inv(omega) @ P
        M3 = np.linalg.inv(M1 + M2)
        
        Sigma_bl = tau_Sigma @ M3
        
        # Posterior returns
        term1 = np.linalg.inv(tau_Sigma) @ mu_market.values
        term2 = P.T @ np.linalg.inv(omega) @ Q
        mu_bl = Sigma_bl @ (term1 + term2)
        
        mu_bl_series = pd.Series(mu_bl, index=mu_market.index)
        Sigma_bl_df = pd.DataFrame(Sigma_bl, index=Sigma.index, columns=Sigma.columns)
        
        return {
            'mu_bl': mu_bl_series,
            'Sigma_bl': Sigma_bl_df,
            'method': 'black_litterman'
        }

# Convenience functions for common optimization tasks
def optimize_portfolio(method: str, mu: pd.Series, Sigma: pd.DataFrame,
                      risk_free_rate: float = 0.02, **kwargs) -> Dict:
    """
    Unified interface for portfolio optimization
    
    Args:
        method: Optimization method ('max_sharpe', 'min_variance', 'risk_parity')
        mu: Expected returns
        Sigma: Covariance matrix
        risk_free_rate: Risk-free rate
        **kwargs: Additional method-specific arguments
        
    Returns:
        Optimization results dictionary
    """
    optimizer = PortfolioOptimizer(risk_free_rate)
    
    if method == 'max_sharpe':
        return optimizer.maximize_sharpe_ratio(mu, Sigma, **kwargs)
    elif method == 'min_variance':
        return optimizer.minimize_variance(Sigma,mu=mu, **kwargs)
    elif method == 'risk_parity':
        return optimizer.risk_parity(Sigma,mu=mu, **kwargs)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

def get_equal_weight_portfolio(assets: List[str]) -> pd.Series:
    """
    Create equal-weight portfolio
    
    Args:
        assets: List of asset names
        
    Returns:
        Equal-weight portfolio series
    """
    n = len(assets)
    return pd.Series(1.0 / n, index=assets)