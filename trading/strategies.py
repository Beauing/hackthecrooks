import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

@dataclass
class TradingSignal:
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0 to 1
    indicators: Dict  # Contains all indicator values
    timestamp: datetime

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        price_diff = pd.Series(prices).diff()
        gain = price_diff.clip(lower=0)
        loss = -price_diff.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    @staticmethod
    def calculate_macd(prices: List[float]) -> Tuple[List[float], List[float]]:
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.tolist(), signal.tolist()

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Tuple[List[float], List[float]]:
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band.tolist(), lower_band.tolist()

    @staticmethod
    def calculate_vwap(prices: List[float], volumes: List[float]) -> float:
        df = pd.DataFrame({'price': prices, 'volume': volumes})
        return (df['price'] * df['volume']).sum() / df['volume'].sum()

    @staticmethod
    def calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
        df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        df['tr0'] = df['high'] - df['low']
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        return df['tr'].rolling(window=period).mean().iloc[-1]

    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
        diff = high - low
        return {
            '0.236': low + diff * 0.236,
            '0.382': low + diff * 0.382,
            '0.500': low + diff * 0.500,
            '0.618': low + diff * 0.618,
            '0.786': low + diff * 0.786
        }

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
                'min_valley_distance': 10,  # minimum bars between valleys
                'max_price_diff': 0.02,     # maximum price difference between valleys
                'breakout_confirmation': 0.01  # price increase needed for confirmation
            },
            'double_top': {
                'min_peak_distance': 10,
                'max_price_diff': 0.02,
                'breakdown_confirmation': 0.01
            }
        }

    def analyze_price_pattern(self, prices: List[float]) -> Optional[Dict]:
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

    def get_advanced_signal(self, 
                          prices: List[float],
                          volumes: List[float],
                          highs: List[float],
                          lows: List[float]) -> TradingSignal:
        
        # Calculate all indicators
        indicators = {}
        
        # EMAs
        for period in self.config['ema_periods']:
            indicators[f'ema_{period}'] = self.indicators.calculate_ema(prices, period)[-1]
        
        # RSI
        indicators['rsi'] = self.indicators.calculate_rsi(prices, self.config['rsi_period'])
        
        # MACD
        macd, signal = self.indicators.calculate_macd(prices)
        indicators['macd'] = macd[-1]
        indicators['macd_signal'] = signal[-1]
        
        # Bollinger Bands
        upper_band, lower_band = self.indicators.calculate_bollinger_bands(prices)
        indicators['bb_upper'] = upper_band[-1]
        indicators['bb_lower'] = lower_band[-1]
        
        # VWAP
        indicators['vwap'] = self.indicators.calculate_vwap(prices[-self.config['vwap_period']:],
                                                          volumes[-self.config['vwap_period']:])
        
        # ATR
        indicators['atr'] = self.indicators.calculate_atr(highs, lows, prices)
        
        # Price patterns
        pattern = self.analyze_price_pattern(prices)
        if pattern:
            indicators['pattern'] = pattern
        
        # Calculate signal scores
        trend_score = self.calculate_trend_score(indicators, prices[-1])
        momentum_score = self.calculate_momentum_score(indicators)
        volatility_score = self.calculate_volatility_score(indicators, prices[-1])
        pattern_score = self.calculate_pattern_score(pattern) if pattern else 0
        
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

    def calculate_trend_score(self, indicators: Dict, current_price: float) -> float:
        score = 0
        
        # EMA alignment
        ema_9 = indicators['ema_9']
        ema_21 = indicators['ema_21']
        ema_50 = indicators['ema_50']
        
        if ema_9 > ema_21 > ema_50:
            score += 0.5
        elif ema_9 < ema_21 < ema_50:
            score -= 0.5
        
        # Price vs VWAP
        if current_price > indicators['vwap']:
            score += 0.3
        else:
            score -= 0.3
        
        return score

    def calculate_momentum_score(self, indicators: Dict) -> float:
        score = 0
        
        # RSI
        if indicators['rsi'] < self.config['rsi_oversold']:
            score += 0.4
        elif indicators['rsi'] > self.config['rsi_overbought']:
            score -= 0.4
            
        # MACD
        if indicators['macd'] > indicators['macd_signal']:
            score += 0.4
        else:
            score -= 0.4
            
        return score

    def calculate_volatility_score(self, indicators: Dict, current_price: float) -> float:
        score = 0
        
        # Bollinger Bands position
        bb_range = indicators['bb_upper'] - indicators['bb_lower']
        if current_price < indicators['bb_lower']:
            score += 0.5
        elif current_price > indicators['bb_upper']:
            score -= 0.5
            
        # ATR consideration
        atr_percent = indicators['atr'] / current_price
        if atr_percent > 0.02:  # High volatility
            score *= 0.8  # Reduce score in high volatility
            
        return score

    def calculate_pattern_score(self, pattern: Dict) -> float:
        if pattern['pattern'] == 'double_bottom':
            return 0.5 * pattern['confidence']
        elif pattern['pattern'] == 'double_top':
            return -0.5 * pattern['confidence']
        return 0.0