# src/feature_engineering/technical_features.py
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple

class TechnicalFeatureEngineer:
    def __init__(self):
        pass
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # VWAP
        df['vwap'] = self.calculate_vwap(df)
        
        # Session ranges
        df['session_range'] = df['high'] - df['low']
        df['avg_session_range'] = df['session_range'].rolling(20).mean()
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Additional indicators
        df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
        df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Ichimoku (simplified)
        df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        
        # Market profile features
        df['value_area_high'] = self.calculate_value_area(df, 'high')
        df['value_area_low'] = self.calculate_value_area(df, 'low')
        
        return df
    
    def calculate_vwap(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_volume = df['volume'].cumsum()
        cumulative_tp_volume = (typical_price * df['volume']).cumsum()
        
        return cumulative_tp_volume / cumulative_volume
    
    def calculate_value_area(self, df: pd.DataFrame, bound: str) -> np.ndarray:
        """Calculate value area high/low"""
        # Implementation using volume profile
        return df[bound].rolling(20).mean()
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(100).mean()
        
        # Historical volatility
        df['hv_10'] = df['returns'].rolling(10).std() * np.sqrt(252)
        df['hv_30'] = df['returns'].rolling(30).std() * np.sqrt(252)
        df['hv_60'] = df['returns'].rolling(60).std() * np.sqrt(252)
        
        # GARCH-like features (simplified)
        df['squared_returns'] = df['returns'] ** 2
        df['avg_squared_returns'] = df['squared_returns'].rolling(20).mean()
        
        return df
    
    def calculate_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        df = df.copy()
        
        # Spread (if bid/ask available)
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread'] = df['ask'] - df['bid']
            df['spread_pct'] = df['spread'] / df['bid'] * 100
            df['avg_spread'] = df['spread'].rolling(20).mean()
        
        # Tick-level features (if data available)
        # Price oscillation
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['momentum_ratio'] = df['momentum'] / df['close'].shift(10)
        
        # Rate of change
        df['roc'] = talib.ROCR(df['close'], timeperiod=10)
        
        return df
    
    def calculate_correlation_features(self, df: pd.DataFrame, 
                                    correlated_symbols: List[str]) -> pd.DataFrame:
        """Calculate correlation with other assets"""
        # Implementation for multi-asset correlation
        pass