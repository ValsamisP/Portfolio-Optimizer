# src/rebalancer.py
"""
Portfolio rebalancing logic including drift analysis, order generation,
and transaction cost optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging

logger = logging.getLogger(__name__)

class PortfolioRebalancer:
    """Handles portfolio rebalancing logic and order generation"""
    
    def __init__(self, min_trade_size: float = 25.0, 
                 transaction_cost_bps: float = 5.0):
        self.min_trade_size = min_trade_size
        self.transaction_cost_bps = transaction_cost_bps
        
    def calculate_current_weights(self, holdings: pd.Series, 
                                prices: pd.Series) -> pd.Series:
        """
        Calculate current portfolio weights from holdings and prices
        
        Args:
            holdings: Current share quantities
            prices: Current share prices
            
        Returns:
            Current portfolio weights
        """
        # Align holdings and prices
        aligned_data = pd.DataFrame({
            'holdings': holdings,
            'prices': prices
        }).fillna(0)
        
        # Calculate market values
        market_values = aligned_data['holdings'] * aligned_data['prices']
        total_value = market_values.sum()
        
        # Calculate weights (handle zero total value)
        if total_value == 0:
            return pd.Series(0.0, index=market_values.index)
        else:
            return market_values / total_value
    
    def analyze_drift(self, current_weights: pd.Series, 
                     target_weights: pd.Series,
                     drift_threshold: float = 0.05) -> Dict[str, pd.Series]:
        """
        Analyze weight drift relative to targets
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            drift_threshold: Threshold for significant drift
            
        Returns:
            Dictionary with drift analysis
        """
        # Align weights
        aligned_weights = pd.DataFrame({
            'current': current_weights,
            'target': target_weights
        }).fillna(0)
        
        # Calculate absolute and relative drift
        absolute_drift = aligned_weights['current'] - aligned_weights['target']
        
        # Relative drift (avoid division by zero)
        relative_drift = pd.Series(0.0, index=aligned_weights.index)
        non_zero_targets = aligned_weights['target'] != 0
        relative_drift[non_zero_targets] = (
            absolute_drift[non_zero_targets] / aligned_weights['target'][non_zero_targets]
        )
        
        # Identify significant drifts
        significant_drift = abs(relative_drift) > drift_threshold
        
        return {
            'absolute_drift': absolute_drift,
            'relative_drift': relative_drift,
            'significant_drift': significant_drift,
            'max_absolute_drift': float(abs(absolute_drift).max()),
            'max_relative_drift': float(abs(relative_drift).max()),
            'assets_over_threshold': significant_drift.sum()
        }
    
    def calculate_rebalance_bands(self, target_weights: pd.Series,
                                 band_width: float = 0.20) -> Dict[str, pd.Series]:
        """
        Calculate rebalancing bands around target weights
        
        Args:
            target_weights: Target portfolio weights
            band_width: Band width as fraction (0.20 = ±20%)
            
        Returns:
            Dictionary with upper and lower bands
        """
        lower_band = target_weights * (1 - band_width)
        upper_band = target_weights * (1 + band_width)
        
        # Ensure bands are within [0, 1]
        lower_band = lower_band.clip(lower=0)
        upper_band = upper_band.clip(upper=1)
        
        return {
            'lower_band': lower_band,
            'upper_band': upper_band,
            'band_width': band_width
        }
    
    def identify_rebalance_needs(self, current_weights: pd.Series,
                               target_weights: pd.Series,
                               rebalance_bands: Dict[str, pd.Series]) -> pd.Series:
        """
        Identify which assets need rebalancing based on drift bands
        
        Args:
            current_weights: Current weights
            target_weights: Target weights
            rebalance_bands: Rebalancing bands
            
        Returns:
            Boolean series indicating which assets need rebalancing
        """
        lower_band = rebalance_bands['lower_band']
        upper_band = rebalance_bands['upper_band']
        
        # Assets are out of band if current weight is outside the bands
        # OR if current weight is zero but target is non-zero (new positions)
        below_band = current_weights < lower_band
        above_band = current_weights > upper_band
        new_positions = (current_weights == 0) & (target_weights > 0)
        
        needs_rebalancing = below_band | above_band | new_positions
        
        return needs_rebalancing
    
    def generate_orders(self, current_holdings: pd.Series,
                       target_weights: pd.Series, 
                       prices: pd.Series,
                       portfolio_value: float,
                       needs_rebalancing: pd.Series) -> pd.DataFrame:
        """
        Generate trading orders for rebalancing
        
        Args:
            current_holdings: Current share quantities
            target_weights: Target portfolio weights
            prices: Current prices
            portfolio_value: Total portfolio value
            needs_rebalancing: Assets that need rebalancing
            
        Returns:
            DataFrame with trading orders
        """
        # Calculate target dollar amounts
        target_values = target_weights * portfolio_value
        current_values = current_holdings * prices
        
        # Calculate required trades (in dollars)
        trade_values = target_values - current_values
        
        # Only trade assets that need rebalancing
        trade_values = trade_values.where(needs_rebalancing, 0)
        
        # Filter out tiny trades
        significant_trades = abs(trade_values) >= self.min_trade_size
        trade_values = trade_values.where(significant_trades, 0)
        
        # Convert dollar amounts to share quantities
        share_quantities = pd.Series(0.0, index=trade_values.index)
        non_zero_prices = prices != 0
        share_quantities[non_zero_prices] = (
            trade_values[non_zero_prices] / prices[non_zero_prices]
        )
        
        # Determine trade direction
        trade_directions = pd.Series('HOLD', index=trade_values.index)
        trade_directions[trade_values > 0] = 'BUY'
        trade_directions[trade_values < 0] = 'SELL'
        
        # Create orders DataFrame
        orders = pd.DataFrame({
            'ticker': trade_values.index,
            'action': trade_directions,
            'quantity': abs(share_quantities).round(3),
            'price': prices.round(4),
            'value_usd': trade_values.round(2),
            'current_weight': (current_values / portfolio_value).round(4),
            'target_weight': target_weights.round(4),
            'weight_diff': ((target_values - current_values) / portfolio_value).round(4)
        })
        
        # Filter out HOLD orders
        orders = orders[orders['action'] != 'HOLD'].reset_index(drop=True)
        
        return orders
    
    def estimate_transaction_costs(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate transaction costs for orders
        
        Args:
            orders: Orders DataFrame
            
        Returns:
            Orders DataFrame with cost estimates
        """
        if orders.empty:
            return orders
        
        # Calculate costs as basis points of trade value
        trade_costs = abs(orders['value_usd']) * (self.transaction_cost_bps / 10000)
        
        # Add different cost components
        orders = orders.copy()
        orders['commission_cost'] = trade_costs * 0.3  # Commission
        orders['bid_ask_spread_cost'] = trade_costs * 0.4  # Bid-ask spread
        orders['market_impact_cost'] = trade_costs * 0.3  # Market impact
        orders['total_cost'] = trade_costs
        orders['cost_bps'] = self.transaction_cost_bps
        
        return orders
    
    def optimize_order_execution(self, orders: pd.DataFrame,
                                max_daily_volume_pct: float = 0.1) -> pd.DataFrame:
        """
        Optimize order execution to minimize market impact
        
        Args:
            orders: Orders DataFrame
            max_daily_volume_pct: Max percentage of daily volume to trade
            
        Returns:
            Orders with execution recommendations
        """
        if orders.empty:
            return orders
        
        orders = orders.copy()
        
        # Categorize orders by size
        small_orders = abs(orders['value_usd']) <= 1000
        medium_orders = (abs(orders['value_usd']) > 1000) & (abs(orders['value_usd']) <= 10000)
        large_orders = abs(orders['value_usd']) > 10000
        
        # Add execution recommendations
        orders['execution_style'] = 'MARKET'  # Default
        orders.loc[medium_orders, 'execution_style'] = 'LIMIT'
        orders.loc[large_orders, 'execution_style'] = 'TWAP'  # Time-weighted average price
        
        # Estimate execution timeframe
        orders['execution_days'] = 1  # Default: same day
        orders.loc[large_orders, 'execution_days'] = 3  # Spread large orders over 3 days
        
        # Priority scoring (higher = more urgent)
        orders['priority_score'] = abs(orders['weight_diff']) * 100
        
        return orders
    
    def create_rebalancing_report(self, current_weights: pd.Series,
                                target_weights: pd.Series,
                                orders: pd.DataFrame,
                                drift_analysis: Dict) -> Dict:
        """
        Create comprehensive rebalancing report
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights
            orders: Generated orders
            drift_analysis: Drift analysis results
            
        Returns:
            Comprehensive rebalancing report
        """
        report = {}
        
        # Portfolio summary
        total_trade_value = orders['value_usd'].abs().sum() if not orders.empty else 0
        total_costs = orders['total_cost'].sum() if 'total_cost' in orders.columns else 0
        
        report['summary'] = {
            'total_trades': len(orders),
            'total_trade_value': float(total_trade_value),
            'total_estimated_costs': float(total_costs),
            'cost_as_pct_of_trades': float(total_costs / total_trade_value * 100) if total_trade_value > 0 else 0,
            'max_weight_drift': drift_analysis['max_absolute_drift'],
            'assets_needing_rebalance': int(drift_analysis['assets_over_threshold'])
        }
        
        # Weight comparison
        weight_comparison = pd.DataFrame({
            'current_weight': current_weights,
            'target_weight': target_weights,
            'absolute_drift': drift_analysis['absolute_drift'],
            'relative_drift_pct': drift_analysis['relative_drift'] * 100
        }).round(4)
        
        report['weight_analysis'] = weight_comparison
        
        # Trade breakdown
        if not orders.empty:
            buy_orders = orders[orders['action'] == 'BUY']
            sell_orders = orders[orders['action'] == 'SELL']
            
            report['trade_breakdown'] = {
                'num_buy_orders': len(buy_orders),
                'num_sell_orders': len(sell_orders),
                'total_buy_value': float(buy_orders['value_usd'].sum()) if not buy_orders.empty else 0,
                'total_sell_value': float(abs(sell_orders['value_usd']).sum()) if not sell_orders.empty else 0,
                'net_cash_flow': float(sell_orders['value_usd'].sum()) if not sell_orders.empty else 0
            }
        else:
            report['trade_breakdown'] = {
                'num_buy_orders': 0,
                'num_sell_orders': 0,
                'total_buy_value': 0,
                'total_sell_value': 0,
                'net_cash_flow': 0
            }
        
        report['orders'] = orders
        
        return report
    
    def full_rebalancing_analysis(self, current_holdings: pd.Series,
                                 target_weights: pd.Series,
                                 prices: pd.Series,
                                 portfolio_value: Optional[float] = None,
                                 drift_threshold: float = 0.05,
                                 band_width: float = 0.20) -> Dict:
        """
        Complete rebalancing analysis workflow
        
        Args:
            current_holdings: Current share quantities
            target_weights: Target portfolio weights
            prices: Current prices
            portfolio_value: Total portfolio value (calculated if None)
            drift_threshold: Drift threshold for analysis
            band_width: Rebalancing band width
            
        Returns:
            Complete rebalancing analysis
        """
        # Calculate current weights and portfolio value
        current_weights = self.calculate_current_weights(current_holdings, prices)
        
        if portfolio_value is None:
            portfolio_value = float((current_holdings * prices).sum())
        
        # Analyze drift
        drift_analysis = self.analyze_drift(current_weights, target_weights, drift_threshold)
        
        # Calculate rebalancing bands
        rebalance_bands = self.calculate_rebalance_bands(target_weights, band_width)
        
        # Identify rebalancing needs
        needs_rebalancing = self.identify_rebalance_needs(
            current_weights, target_weights, rebalance_bands
        )
        
        # Generate orders
        orders = self.generate_orders(
            current_holdings, target_weights, prices, portfolio_value, needs_rebalancing
        )
        
        # Add cost estimates and execution optimization
        if not orders.empty:
            orders = self.estimate_transaction_costs(orders)
            orders = self.optimize_order_execution(orders)
        
        # Create comprehensive report
        report = self.create_rebalancing_report(
            current_weights, target_weights, orders, drift_analysis
        )
        
        # Add additional analysis components
        report['rebalance_bands'] = rebalance_bands
        report['needs_rebalancing'] = needs_rebalancing
        report['portfolio_value'] = portfolio_value
        
        return report

# Convenience functions
def simple_rebalance_check(current_holdings: pd.Series, target_weights: pd.Series,
                          prices: pd.Series, band_width: float = 0.20) -> bool:
    """
    Quick check if portfolio needs rebalancing
    
    Args:
        current_holdings: Current holdings
        target_weights: Target weights
        prices: Current prices
        band_width: Rebalancing band width
        
    Returns:
        True if rebalancing is needed
    """
    rebalancer = PortfolioRebalancer()
    current_weights = rebalancer.calculate_current_weights(current_holdings, prices)
    
    drift_analysis = rebalancer.analyze_drift(current_weights, target_weights)
    rebalance_bands = rebalancer.calculate_rebalance_bands(target_weights, band_width)
    needs_rebalancing = rebalancer.identify_rebalance_needs(
        current_weights, target_weights, rebalance_bands
    )
    
    return needs_rebalancing.any()

def calculate_turnover(previous_weights: pd.Series, current_weights: pd.Series) -> float:
    """
    Calculate portfolio turnover between two periods
    
    Args:
        previous_weights: Previous period weights
        current_weights: Current period weights
        
    Returns:
        Turnover as percentage
    """
    # Align weights
    aligned_data = pd.DataFrame({
        'previous': previous_weights,
        'current': current_weights
    }).fillna(0)
    
    # Turnover = sum of absolute weight changes / 2
    weight_changes = abs(aligned_data['current'] - aligned_data['previous'])
    turnover = weight_changes.sum() / 2
    
    return float(turnover * 100)  # Return as percentage