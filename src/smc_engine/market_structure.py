# src/smc_engine/market_structure.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class SwingPoint:
    price: float
    timestamp: pd.Timestamp
    index: int
    type: str  # 'high' or 'low'
    strength: float
    volume: Optional[float] = None

@dataclass
class MarketStructure:
    trend: str  # 'bullish', 'bearish', 'ranging'
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    structure_levels: Dict[str, float]
    breaks: List[Dict]

class MarketStructureAnalyzer:
    def __init__(self, window: int = 5, strength_threshold: float = 0.3):
        self.window = window
        self.strength_threshold = strength_threshold
        self.swing_highs = []
        self.swing_lows = []
        
    def analyze(self, df: pd.DataFrame) -> MarketStructure:
        """Analyze market structure from OHLCV data"""
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # Identify swing points
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)
        
        # Determine trend
        trend = self._determine_trend(swing_highs, swing_lows)
        
        # Calculate structure levels
        structure_levels = self._calculate_structure_levels(swing_highs, swing_lows)
        
        # Detect breaks
        breaks = self._detect_breaks(df, swing_highs, swing_lows)
        
        return MarketStructure(
            trend=trend,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            structure_levels=structure_levels,
            breaks=breaks
        )
    
    def _find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing highs in the data"""
        swing_highs = []
        highs = df['high'].values
        prices = df['close'].values if 'close' in df.columns else highs
        
        for i in range(self.window, len(df) - self.window):
            if highs[i] == max(highs[i - self.window:i + self.window + 1]):
                # Check if it's a significant swing
                strength = self._calculate_swing_strength(df, i, 'high')
                if strength >= self.strength_threshold:
                    swing_highs.append(
                        SwingPoint(
                            price=highs[i],
                            timestamp=df.index[i],
                            index=i,
                            type='high',
                            strength=strength,
                            volume=df.iloc[i]['volume'] if 'volume' in df.columns else None
                        )
                    )
        
        return swing_highs
    
    def _find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing lows in the data"""
        swing_lows = []
        lows = df['low'].values
        prices = df['close'].values if 'close' in df.columns else lows
        
        for i in range(self.window, len(df) - self.window):
            if lows[i] == min(lows[i - self.window:i + self.window + 1]):
                strength = self._calculate_swing_strength(df, i, 'low')
                if strength >= self.strength_threshold:
                    swing_lows.append(
                        SwingPoint(
                            price=lows[i],
                            timestamp=df.index[i],
                            index=i,
                            type='low',
                            strength=strength,
                            volume=df.iloc[i]['volume'] if 'volume' in df.columns else None
                        )
                    )
        
        return swing_lows
    
    def _calculate_swing_strength(self, df: pd.DataFrame, index: int, 
                                type: str) -> float:
        """Calculate strength of a swing point"""
        price = df.iloc[index]['high'] if type == 'high' else df.iloc[index]['low']
        volume = df.iloc[index]['volume'] if 'volume' in df.columns else None
        
        # Calculate strength based on volume and price action
        strength = 0.5  # Base strength
        
        if volume is not None:
            avg_volume = df['volume'].rolling(20).mean().iloc[index]
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                strength += min(volume_ratio * 0.3, 0.3)
        
        # Check if price bounced strongly from this level
        if type == 'high':
            future_lows = df['low'].iloc[index:index + 5].min() if index < len(df) - 5 else price
            bounce = (price - future_lows) / price if price > 0 else 0
            strength += min(bounce * 2, 0.2)
        else:
            future_highs = df['high'].iloc[index:index + 5].max() if index < len(df) - 5 else price
            bounce = (future_highs - price) / price if price > 0 else 0
            strength += min(bounce * 2, 0.2)
        
        return min(strength, 1.0)
    
    def _determine_trend(self, swing_highs: List[SwingPoint], 
                        swing_lows: List[SwingPoint]) -> str:
        """Determine market trend"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'ranging'
        
        # Check recent swing points
        recent_highs = swing_highs[-3:]
        recent_lows = swing_lows[-3:]
        
        high_trend = all(recent_highs[i].price < recent_highs[i+1].price 
                        for i in range(len(recent_highs) - 1))
        low_trend = all(recent_lows[i].price < recent_lows[i+1].price 
                       for i in range(len(recent_lows) - 1))
        
        if high_trend and low_trend:
            return 'bullish'
        elif not high_trend and not low_trend:
            return 'bearish'
        else:
            return 'ranging'
    
    def _calculate_structure_levels(self, swing_highs: List[SwingPoint], 
                                   swing_lows: List[SwingPoint]) -> Dict[str, float]:
        """Calculate key structure levels"""
        levels = {}
        
        if swing_highs:
            # Recent swing high
            levels['recent_high'] = swing_highs[-1].price
            
            # Highest high
            levels['highest_high'] = max(sh.price for sh in swing_highs)
            
            # Average high
            levels['avg_high'] = sum(sh.price for sh in swing_highs) / len(swing_highs)
        
        if swing_lows:
            # Recent swing low
            levels['recent_low'] = swing_lows[-1].price
            
            # Lowest low
            levels['lowest_low'] = min(sl.price for sl in swing_lows)
            
            # Average low
            levels['avg_low'] = sum(sl.price for sl in swing_lows) / len(swing_lows)
        
        return levels
    
    def _detect_breaks(self, df: pd.DataFrame, swing_highs: List[SwingPoint], 
                      swing_lows: List[SwingPoint]) -> List[Dict]:
        """Detect breaks of structure"""
        breaks = []
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            last_high = swing_highs[-1]
            prev_high = swing_highs[-2]
            last_low = swing_lows[-1]
            prev_low = swing_lows[-2]
            
            current_price = df.iloc[-1]['close']
            
            # Break of structure (BOS)
            if current_price > last_high.price and last_high.price > prev_high.price:
                breaks.append({
                    'type': 'bullish_bos',
                    'price': current_price,
                    'timestamp': df.index[-1],
                    'strength': 0.8
                })
            elif current_price < last_low.price and last_low.price < prev_low.price:
                breaks.append({
                    'type': 'bearish_bos',
                    'price': current_price,
                    'timestamp': df.index[-1],
                    'strength': 0.8
                })
            
            # Change of Character (CHOCH)
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                # Check for trend reversal signs
                if (last_high.price < prev_high.price and 
                    last_low.price < prev_low.price and
                    current_price < last_low.price):
                    breaks.append({
                        'type': 'bearish_choch',
                        'price': current_price,
                        'timestamp': df.index[-1],
                        'strength': 0.9
                    })
                elif (last_high.price > prev_high.price and 
                      last_low.price > prev_low.price and
                      current_price > last_high.price):
                    breaks.append({
                        'type': 'bullish_choch',
                        'price': current_price,
                        'timestamp': df.index[-1],
                        'strength': 0.9
                    })
        
        return breaks