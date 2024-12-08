import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        """Calculate EMA with safety checks"""
        try:
            if len(prices) < period:
                return []
            return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()
        except Exception as e:
            print(f"Error calculating EMA: {str(e)}")
            return []

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI with safety checks"""
        try:
            if len(prices) < period + 1:
                return None
            price_diff = pd.Series(prices).diff()
            gain = price_diff.clip(lower=0)
            loss = -price_diff.clip(upper=0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            if avg_loss.iloc[-1] == 0:
                return 100.0
            rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
            return 100 - (100 / (1 + rs))
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return None

    @staticmethod
    def calculate_macd(prices: List[float]) -> Tuple[List[float], List[float]]:
        """Calculate MACD with safety checks"""
        try:
            if len(prices) < 26:
                return [], []
            exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
            exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            return macd.tolist(), signal.tolist()
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
            return [], []

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Tuple[List[float], List[float]]:
        """Calculate Bollinger Bands with safety checks"""
        try:
            if len(prices) < period:
                return [], []
            series = pd.Series(prices)
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            return upper_band.tolist(), lower_band.tolist()
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {str(e)}")
            return [], []

    @staticmethod
    def calculate_vwap(prices: List[float], volumes: List[float]) -> Optional[float]:
        """Calculate VWAP with safety checks"""
        try:
            if len(prices) != len(volumes) or not prices:
                return None
            df = pd.DataFrame({'price': prices, 'volume': volumes})
            return (df['price'] * df['volume']).sum() / df['volume'].sum()
        except Exception as e:
            print(f"Error calculating VWAP: {str(e)}")
            return None

    @staticmethod
    def calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]:
        """Calculate ATR with safety checks"""
        try:
            if len(high) < period or len(low) < period or len(close) < period:
                return None
            df = pd.DataFrame({'high': high, 'low': low, 'close': close})
            df['tr0'] = df['high'] - df['low']
            df['tr1'] = abs(df['high'] - df['close'].shift())
            df['tr2'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            return df['tr'].rolling(window=period).mean().iloc[-1]
        except Exception as e:
            print(f"Error calculating ATR: {str(e)}")
            return None

    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci levels with safety checks"""
        try:
            if high <= low:
                return {}
            diff = high - low
            return {
                '0.236': low + diff * 0.236,
                '0.382': low + diff * 0.382,
                '0.500': low + diff * 0.500,
                '0.618': low + diff * 0.618,
                '0.786': low + diff * 0.786
            }
        except Exception as e:
            print(f"Error calculating Fibonacci levels: {str(e)}")
            return {}