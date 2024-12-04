import numpy as np
from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta

class EnhancedTokenAnalysis:
    def __init__(self):
        self.metrics = {
            'whale_threshold': 0.05,  # 5% of total supply
            'min_tx_count': 100,      # Minimum transactions per day
            'max_wallet_concentration': 0.2,  # Max 20% held by top 10 wallets
            'volume_consistency_threshold': 0.3  # Max 30% volume variance
        }

    def analyze_wallet_distribution(self, holders: List[Dict]) -> Dict:
        """Analyze wallet distribution and concentration"""
        total_supply = sum(h['balance'] for h in holders)
        sorted_holders = sorted(holders, key=lambda x: x['balance'], reverse=True)
        top_10_holdings = sum(h['balance'] for h in sorted_holders[:10])
        
        # Calculate Gini coefficient for wallet distribution
        cumulative_balance = np.cumsum([h['balance'] for h in sorted_holders])
        gini = 1 - 2 * np.trapz(cumulative_balance) / (cumulative_balance[-1] * len(holders))
        
        return {
            'wallet_concentration': top_10_holdings / total_supply,
            'gini_coefficient': gini,
            'unique_holders': len(holders),
            'whale_count': sum(1 for h in holders if h['balance'] / total_supply > self.metrics['whale_threshold'])
        }

    def analyze_volume_profile(self, volume_data: List[Dict]) -> Dict:
        """Enhanced volume analysis"""
        volumes = [v['volume'] for v in volume_data]
        timestamps = [v['timestamp'] for v in volume_data]
        df = pd.DataFrame({'volume': volumes, 'timestamp': timestamps})
        
        # Volume consistency
        volume_std = np.std(volumes)
        volume_mean = np.mean(volumes)
        volume_cv = volume_std / volume_mean if volume_mean > 0 else float('inf')
        
        # Volume trend
        df['volume_ma'] = df['volume'].rolling(window=24).mean()
        volume_trend = (df['volume_ma'].iloc[-1] / df['volume_ma'].iloc[0]) - 1 if len(df) > 24 else 0
        
        # Unusual volume spikes
        volume_zscore = (volumes[-1] - volume_mean) / volume_std if volume_std > 0 else 0
        
        return {
            'volume_consistency': volume_cv,
            'volume_trend': volume_trend,
            'volume_zscore': volume_zscore,
            'avg_daily_volume': volume_mean * 24,
            'volume_profile': self.calculate_volume_profile(df)
        }

    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed volume profile"""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_profile = df.groupby('hour')['volume'].mean()
        
        return {
            'peak_hour': hourly_profile.idxmax(),
            'low_hour': hourly_profile.idxmin(),
            'volume_distribution': hourly_profile.to_dict()
        }

    def analyze_price_action(self, price_data: List[Dict]) -> Dict:
        """Enhanced price analysis"""
        prices = [p['price'] for p in price_data]
        df = pd.DataFrame({'price': prices})
        
        # Calculate various technical indicators
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        df['std_20'] = df['price'].rolling(window=20).std()
        
        # Price momentum
        momentum = (df['price'].iloc[-1] / df['price'].iloc[-20] - 1) if len(df) >= 20 else 0
        
        # Volatility measures
        daily_returns = df['price'].pct_change()
        volatility = daily_returns.std() * np.sqrt(365)
        
        # Price trend strength
        if len(df) >= 50:
            trend_strength = abs(df['sma_20'].iloc[-1] / df['sma_50'].iloc[-1] - 1)
        else:
            trend_strength = 0
            
        return {
            'momentum': momentum,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'price_efficiency': self.calculate_price_efficiency(prices),
            'fractal_dimension': self.calculate_fractal_dimension(prices)
        }

    def calculate_price_efficiency(self, prices: List[float]) -> float:
        """Calculate price efficiency ratio"""
        if len(prices) < 2:
            return 0
            
        direct_distance = abs(prices[-1] - prices[0])
        path_distance = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
        
        return direct_distance / path_distance if path_distance > 0 else 0

    def calculate_fractal_dimension(self, prices: List[float], max_lag: int = 20) -> float:
        """Calculate fractal dimension using Hurst exponent"""
        if len(prices) < max_lag:
            return 0
            
        lags = range(2, min(max_lag, len(prices)))
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        
        if not all(tau):
            return 0
            
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = reg[0]
        
        return 2 - hurst  # Convert Hurst exponent to fractal dimension

    def detect_wash_trading(self, trades: List[Dict]) -> Dict:
        """Detect potential wash trading patterns"""
        df = pd.DataFrame(trades)
        
        # Look for circular trading patterns
        circular_trades = 0
        suspicious_volumes = 0
        
        if not df.empty:
            # Group trades by wallet pairs
            wallet_pairs = df.groupby(['from_wallet', 'to_wallet']).size()
            reciprocal_pairs = wallet_pairs[wallet_pairs > 5].sum()  # Suspicious if >5 trades between same wallets
            
            # Check for uniform trade sizes
            trade_sizes = df['volume'].value_counts()
            suspicious_volumes = trade_sizes[trade_sizes > 3].sum()  # Suspicious if same size appears >3 times
            
            circular_trades = reciprocal_pairs
            
        return {
            'circular_trades': circular_trades,
            'suspicious_volumes': suspicious_volumes,
            'wash_trading_score': (circular_trades + suspicious_volumes) / len(trades) if trades else 0
        }

    def analyze_token_distribution(self, token_data: Dict) -> Dict:
        """Analyze token distribution and potential manipulation"""
        # Calculate metrics for token distribution
        supply_distribution = {
            'circulating_ratio': token_data['circulating_supply'] / token_data['total_supply'],
            'locked_ratio': token_data.get('locked_supply', 0) / token_data['total_supply'],
            'burn_ratio': token_data.get('burned_supply', 0) / token_data['total_supply']
        }
        
        # Risk metrics
        risk_metrics = {
            'high_concentration': supply_distribution['locked_ratio'] > 0.5,
            'low_circulation': supply_distribution['circulating_ratio'] < 0.2,
            'suspicious_burns': supply_distribution['burn_ratio'] > 0.9
        }
        
        return {
            'distribution': supply_distribution,
            'risks': risk_metrics,
            'risk_score': sum(1 for v in risk_metrics.values() if v) / len(risk_metrics)
        }