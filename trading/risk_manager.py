import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

@dataclass
class Position:
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None

class RiskManager:
    def __init__(self):
        self.config = {
            'max_position_size': 0.1,  # Maximum 10% of portfolio per trade
            'max_daily_loss': 0.05,    # Maximum 5% daily loss
            'trailing_stop_pct': 0.02,  # 2% trailing stop
            'min_risk_reward': 2.0,    # Minimum risk/reward ratio
            'max_correlation': 0.7,     # Maximum correlation between positions
            'max_drawdown': 0.15,      # Maximum drawdown before reducing position size
            'vol_adjustment': True,     # Adjust position size based on volatility
        }
        
        self.positions: List[Position] = []
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'current_drawdown': 0,
            'peak_balance': 0,
            'last_reset': datetime.now()
        }
        self.trade_history: List[Dict] = []

    def calculate_position_size(self, 
                              wallet_balance: float,
                              signal_confidence: float,
                              volatility: float,
                              current_exposure: float) -> float:
        """Calculate safe position size based on multiple factors"""
        # Base position size
        base_size = wallet_balance * self.config['max_position_size']
        
        # Adjust for confidence
        confidence_adj = base_size * min(signal_confidence, 1.0)
        
        # Adjust for volatility
        vol_adj = confidence_adj
        if self.config['vol_adjustment']:
            vol_multiplier = 1 - (volatility - 0.02) / 0.02  # Normalize around 2% volatility
            vol_adj = confidence_adj * max(0.2, min(1.0, vol_multiplier))
        
        # Adjust for drawdown
        if self.daily_stats['current_drawdown'] > self.config['max_drawdown']:
            drawdown_factor = 1 - (self.daily_stats['current_drawdown'] / self.config['max_drawdown'])
            vol_adj *= drawdown_factor
        
        # Adjust for current exposure
        remaining_exposure = 1 - (current_exposure / wallet_balance)
        vol_adj *= remaining_exposure
        
        return vol_adj

    def calculate_stop_loss(self,
                          entry_price: float,
                          atr: float,
                          pattern_support: Optional[float] = None) -> float:
        """Calculate stop loss price using ATR and support levels"""
        # ATR-based stop loss
        atr_stop = entry_price - (atr * 2)
        
        # If we have a pattern-based support level, use the higher of the two
        if pattern_support:
            return max(atr_stop, pattern_support)
        
        return atr_stop

    def calculate_take_profit(self,
                            entry_price: float,
                            stop_loss: float,
                            min_rr: float = None) -> float:
        """Calculate take profit level ensuring minimum risk/reward ratio"""
        if min_rr is None:
            min_rr = self.config['min_risk_reward']
            
        risk = entry_price - stop_loss
        return entry_price + (risk * min_rr)

    def update_trailing_stop(self,
                           position: Position,
                           current_price: float) -> float:
        """Update trailing stop loss price"""
        if position.trailing_stop is None:
            position.trailing_stop = position.stop_loss
            
        theoretical_stop = current_price * (1 - self.config['trailing_stop_pct'])
        
        if theoretical_stop > position.trailing_stop:
            return theoretical_stop
            
        return position.trailing_stop

    def check_position_exit(self,
                          position: Position,
                          current_price: float,
                          current_time: datetime) -> Dict:
        """Check if position should be exited"""
        pnl = (current_price - position.entry_price) * position.size
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # Update trailing stop
        current_stop = self.update_trailing_stop(position, current_price)
        
        should_exit = False
        exit_reason = None
        
        # Check stop loss
        if current_price <= current_stop:
            should_exit = True
            exit_reason = 'stop_loss'
        
        # Check take profit
        elif current_price >= position.take_profit:
            should_exit = True
            exit_reason = 'take_profit'
        
        # Check time-based exit (24h max hold time)
        elif (current_time - position.entry_time).total_seconds() > 86400:
            should_exit = True
            exit_reason = 'time_exit'
            
        return {
            'should_exit': should_exit,
            'exit_reason': exit_reason,
            'current_pnl': pnl,
            'pnl_percentage': pnl_pct,
            'current_stop': current_stop
        }

    def update_daily_stats(self, trade_result: Dict):
        """Update daily trading statistics"""
        self.daily_stats['total_trades'] += 1
        self.daily_stats['total_pnl'] += trade_result['pnl']
        
        if trade_result['pnl'] > 0:
            self.daily_stats['winning_trades'] += 1
            
        # Update peak balance and drawdown
        if self.daily_stats['total_pnl'] > self.daily_stats['peak_balance']:
            self.daily_stats['peak_balance'] = self.daily_stats['total_pnl']
        else:
            current_drawdown = (self.daily_stats['peak_balance'] - self.daily_stats['total_pnl']) / self.daily_stats['peak_balance']
            self.daily_stats['current_drawdown'] = max(self.daily_stats['current_drawdown'], current_drawdown)
        
        # Record trade in history
        self.trade_history.append({
            'timestamp': datetime.now(),
            'type': trade_result['type'],
            'entry_price': trade_result['entry_price'],
            'exit_price': trade_result['exit_price'],
            'size': trade_result['size'],
            'pnl': trade_result['pnl'],
            'exit_reason': trade_result['exit_reason']
        })

    def reset_daily_stats(self):
        """Reset daily statistics (called at start of each trading day)"""
        if (datetime.now() - self.daily_stats['last_reset']).days >= 1:
            self.daily_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0,
                'current_drawdown': 0,
                'peak_balance': 0,
                'last_reset': datetime.now()
            }

    def check_portfolio_correlation(self, positions: List[Position]) -> float:
        """Calculate correlation between positions to maintain diversification"""
        if len(positions) < 2:
            return 0
            
        # Extract price series for each position
        price_series = []
        for pos in positions:
            # This would need to be implemented to get historical prices for the asset
            # For now, returning a placeholder
            return 0.5

    def can_open_position(self, 
                         wallet_balance: float,
                         proposed_position_size: float) -> Dict:
        """Check if new position can be opened based on risk management rules"""
        
        # Check daily loss limit
        if abs(self.daily_stats['total_pnl']) >= wallet_balance * self.config['max_daily_loss']:
            return {
                'can_open': False,
                'reason': 'daily_loss_limit_reached'
            }
            
        # Check maximum drawdown
        if self.daily_stats['current_drawdown'] >= self.config['max_drawdown']:
            return {
                'can_open': False,
                'reason': 'max_drawdown_reached'
            }
            
        # Check position size limits
        current_exposure = sum(pos.size for pos in self.positions)
        if (current_exposure + proposed_position_size) > wallet_balance * self.config['max_position_size']:
            return {
                'can_open': False,
                'reason': 'position_size_limit_reached'
            }
            
        # Check correlation if other positions exist
        if len(self.positions) > 0:
            correlation = self.check_portfolio_correlation(self.positions)
            if correlation > self.config['max_correlation']:
                return {
                    'can_open': False,
                    'reason': 'correlation_too_high'
                }
                
        return {
            'can_open': True,
            'reason': None
        }

    def get_risk_adjusted_parameters(self,
                                   entry_price: float,
                                   wallet_balance: float,
                                   signal_confidence: float,
                                   volatility: float,
                                   support_level: Optional[float] = None) -> Dict:
        """Get complete risk-adjusted trading parameters for a new position"""
        
        current_exposure = sum(pos.size for pos in self.positions)
        
        position_size = self.calculate_position_size(
            wallet_balance,
            signal_confidence,
            volatility,
            current_exposure
        )
        
        stop_loss = self.calculate_stop_loss(
            entry_price,
            volatility,
            support_level
        )
        
        take_profit = self.calculate_take_profit(
            entry_price,
            stop_loss
        )
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': (take_profit - entry_price) / (entry_price - stop_loss)
        }

    def validate_trade(self,
                      signal: str,
                      confidence: float,
                      price: float,
                      wallet_balance: float,
                      volatility: float) -> bool:
        """Final validation of trade based on all risk parameters"""
        
        # Reset daily stats if needed
        self.reset_daily_stats()
        
        # Get risk-adjusted parameters
        params = self.get_risk_adjusted_parameters(
            price,
            wallet_balance,
            confidence,
            volatility
        )
        
        # Check if position can be opened
        position_check = self.can_open_position(wallet_balance, params['position_size'])
        if not position_check['can_open']:
            return False
            
        # Validate risk-reward ratio
        if params['risk_reward_ratio'] < self.config['min_risk_reward']:
            return False
            
        return True