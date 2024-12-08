import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from trading.indicators import TechnicalIndicators

@dataclass
class TradingSignal:
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0 to 1
    indicators: Dict  # Contains all indicator values
    timestamp: datetime

class AdvancedTradingStrategy:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.config = {
            'ema_periods': [9, 21, 50],
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_periods': [12, 26, 9],
            'bb_period': 20,
            'atr_period': 14,
            'vwap_period': 24,
        }
        self.pattern_rules = self.initialize_pattern_rules()

    def initialize_pattern_rules(self) -> Dict:
        return {
            'double_bottom': {
                'min_valley_distance': 10,
                'max_price_diff': 0.02,
                'breakout_confirmation': 0.01
            },
            'double_top': {
                'min_peak_distance': 10,
                'max_price_diff': 0.02,
                'breakdown_confirmation': 0.01
            }
        }

    def analyze_price_pattern(self, prices: List[float]) -> Optional[Dict]:
        """Analyze price patterns with error handling"""
        try:
            if len(prices) < 10:  # Minimum required for pattern analysis
                return None

            df = pd.DataFrame({'price': prices})
            
            # Identify local minima and maxima
            df['min'] = df['price'].rolling(window=5, center=True).min() == df['price']
            df['max'] = df['price'].rolling(window=5, center=True).max() == df['price']
            
            # Check for double bottom
            valleys = df[df['min']]['price'].tolist()
            if len(valleys) >= 2:
                price_diff = abs(valleys[-1] - valleys[-2]) / valleys[-2]
                if price_diff <= self.pattern_rules['double_bottom']['max_price_diff']:
                    return {'pattern': 'double_bottom', 'confidence': 1 - price_diff}
            
            # Check for double top
            peaks = df[df['max']]['price'].tolist()
            if len(peaks) >= 2:
                price_diff = abs(peaks[-1] - peaks[-2]) / peaks[-2]
                if price_diff <= self.pattern_rules['double_top']['max_price_diff']:
                    return {'pattern': 'double_top', 'confidence': 1 - price_diff}
            
            return None
            
        except Exception as e:
            print(f"Error in pattern analysis: {str(e)}")
            return None

    def get_advanced_signal(self, prices: List[float], volumes: List[float], 
                        highs: List[float], lows: List[float]) -> TradingSignal:
        """Get advanced trading signal with error handling"""
        try:
            # Validate input data
            if len(prices) < max(self.config['ema_periods']):
                return TradingSignal(
                    signal="HOLD",
                    confidence=0,
                    indicators={},
                    timestamp=datetime.now()
                )

            indicators = {}
            
            try:
                # EMAs
                for period in self.config['ema_periods']:
                    if len(prices) >= period:
                        ema = self.indicators.calculate_ema(prices, period)
                        indicators[f'ema_{period}'] = ema[-1] if len(ema) > 0 else None
                    else:
                        indicators[f'ema_{period}'] = None
                
                # RSI
                if len(prices) >= self.config['rsi_period']:
                    indicators['rsi'] = self.indicators.calculate_rsi(prices, self.config['rsi_period'])
                else:
                    indicators['rsi'] = None
                
                # MACD
                if len(prices) >= 26:  # Minimum length for MACD
                    macd, signal = self.indicators.calculate_macd(prices)
                    indicators['macd'] = macd[-1] if len(macd) > 0 else None
                    indicators['macd_signal'] = signal[-1] if len(signal) > 0 else None
                else:
                    indicators['macd'] = None
                    indicators['macd_signal'] = None
                
                # Bollinger Bands
                if len(prices) >= 20:  # Minimum length for Bollinger Bands
                    upper_band, lower_band = self.indicators.calculate_bollinger_bands(prices)
                    indicators['bb_upper'] = upper_band[-1] if len(upper_band) > 0 else None
                    indicators['bb_lower'] = lower_band[-1] if len(lower_band) > 0 else None
                else:
                    indicators['bb_upper'] = None
                    indicators['bb_lower'] = None
                
                # VWAP
                vwap_period = min(self.config['vwap_period'], len(prices))
                if vwap_period > 0 and len(volumes) >= vwap_period:
                    indicators['vwap'] = self.indicators.calculate_vwap(
                        prices[-vwap_period:],
                        volumes[-vwap_period:]
                    )
                else:
                    indicators['vwap'] = None
                
                # ATR
                if len(highs) >= self.config.get('atr_period', 14):
                    indicators['atr'] = self.indicators.calculate_atr(highs, lows, prices)
                else:
                    indicators['atr'] = None
                
                # Price patterns
                pattern = self.analyze_price_pattern(prices)
                if pattern:
                    indicators['pattern'] = pattern
                
            except Exception as e:
                print(f"Error calculating indicators: {str(e)}")
                return TradingSignal(
                    signal="HOLD",
                    confidence=0,
                    indicators={},
                    timestamp=datetime.now()
                )

            # Check if we have enough valid indicators
            valid_indicators = sum(1 for v in indicators.values() if v is not None)
            min_required = len(self.config['ema_periods']) + 3  # EMAs + RSI + MACD + BB
            
            if valid_indicators < min_required:
                return TradingSignal(
                    signal="HOLD",
                    confidence=0,
                    indicators=indicators,
                    timestamp=datetime.now()
                )

            # Calculate scores
            try:
                trend_score = self.calculate_trend_score(indicators, prices[-1])
                momentum_score = self.calculate_momentum_score(indicators)
                volatility_score = self.calculate_volatility_score(indicators, prices[-1])
                pattern_score = self.calculate_pattern_score(pattern) if pattern else 0
            except Exception as e:
                print(f"Error calculating scores: {str(e)}")
                return TradingSignal(
                    signal="HOLD",
                    confidence=0,
                    indicators=indicators,
                    timestamp=datetime.now()
                )

            # Weighted combination of scores
            final_score = (
                trend_score * 0.35 +
                momentum_score * 0.30 +
                volatility_score * 0.20 +
                pattern_score * 0.15
            )
            
            # Determine signal and confidence
            signal = "HOLD"
            if final_score > 0.2:
                signal = "BUY"
            elif final_score < -0.2:
                signal = "SELL"
            
            return TradingSignal(
                signal=signal,
                confidence=abs(final_score),
                indicators=indicators,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Critical error in get_advanced_signal: {str(e)}")
            return TradingSignal(
                signal="HOLD",
                confidence=0,
                indicators={},
                timestamp=datetime.now()
            )

    def calculate_trend_score(self, indicators: Dict, current_price: float) -> float:
        """Calculate trend score with safety checks"""
        try:
            score = 0
            
            # EMA alignment check
            if all(indicators.get(f'ema_{period}') for period in self.config['ema_periods']):
                ema_9 = indicators['ema_9']
                ema_21 = indicators['ema_21']
                ema_50 = indicators['ema_50']
                
                if ema_9 > ema_21 > ema_50:
                    score += 0.5
                elif ema_9 < ema_21 < ema_50:
                    score -= 0.5
            
            # Price vs VWAP
            vwap = indicators.get('vwap')
            if vwap is not None and current_price > 0:
                if current_price > vwap:
                    score += 0.3
                else:
                    score -= 0.3
            
            return score
            
        except Exception as e:
            print(f"Error calculating trend score: {str(e)}")
            return 0.0

    def calculate_momentum_score(self, indicators: Dict) -> float:
        """Calculate momentum score with safety checks"""
        try:
            score = 0
            
            # RSI check
            rsi = indicators.get('rsi')
            if rsi is not None:
                if rsi < self.config['rsi_oversold']:
                    score += 0.4
                elif rsi > self.config['rsi_overbought']:
                    score -= 0.4
                
            # MACD check
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    score += 0.4
                else:
                    score -= 0.4
                
            return score
            
        except Exception as e:
            print(f"Error calculating momentum score: {str(e)}")
            return 0.0

    def calculate_volatility_score(self, indicators: Dict, current_price: float) -> float:
        """Calculate volatility score with safety checks"""
        try:
            score = 0
            
            # Bollinger Bands position
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            
            if all(x is not None for x in [bb_upper, bb_lower, current_price]):
                bb_range = bb_upper - bb_lower
                if current_price < bb_lower:
                    score += 0.5
                elif current_price > bb_upper:
                    score -= 0.5
                
            # ATR consideration
            atr = indicators.get('atr')
            if atr is not None and current_price > 0:
                atr_percent = atr / current_price
                if atr_percent > 0.02:  # High volatility
                    score *= 0.8  # Reduce score in high volatility
                
            return score
            
        except Exception as e:
            print(f"Error calculating volatility score: {str(e)}")
            return 0.0

    def calculate_pattern_score(self, pattern: Optional[Dict]) -> float:
        """Calculate pattern score with safety checks"""
        try:
            if not pattern:
                return 0.0
                
            pattern_type = pattern.get('pattern')
            confidence = pattern.get('confidence', 0.0)
            
            if pattern_type == 'double_bottom':
                return 0.5 * confidence
            elif pattern_type == 'double_top':
                return -0.5 * confidence
                
            return 0.0
            
        except Exception as e:
            print(f"Error calculating pattern score: {str(e)}")
            return 0.0