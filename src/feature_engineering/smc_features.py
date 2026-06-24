# src/feature_engineering/smc_features.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import talib

@dataclass
class OrderBlock:
    start_price: float
    end_price: float
    start_time: datetime
    end_time: datetime
    type: str  # 'bullish' or 'bearish'
    strength: float
    mitigated: bool
    volume: Optional[float] = None

@dataclass
class FVG:
    top: float
    bottom: float
    start_time: datetime
    end_time: datetime
    type: str  # 'bullish' or 'bearish'
    fill_percentage: float
    premium_zone: bool

@dataclass
class LiquidityZone:
    price_level: float
    type: str  # 'buy_sweep' or 'sell_sweep'
    strength: float
    timestamp: datetime
    pool_size: Optional[float] = None

class SMCFeatureEngineer:
    def __init__(self):
        self.order_blocks = []
        self.fvgs = []
        self.liquidity_zones = []
        self.market_structure = {}
    
    def calculate_market_structure(self, df: pd.DataFrame) -> Dict:
        """Calculate market structure: HH, HL, LH, LL"""
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Identify swing points
        swing_highs = []
        swing_lows = []
        window = 5  # Lookback window
        
        for i in range(window, len(df) - window):
            # Swing high
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append({
                    'index': i,
                    'price': highs[i],
                    'timestamp': df.index[i]
                })
            # Swing low
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append({
                    'index': i,
                    'price': lows[i],
                    'timestamp': df.index[i]
                })
        
        # Determine structure
        structure = {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'trend': self._determine_trend(swing_highs, swing_lows)
        }
        
        return structure
    
    def _determine_trend(self, swing_highs: List, swing_lows: List) -> str:
        """Determine if trend is bullish, bearish, or ranging"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'ranging'
        
        # Check for higher highs and higher lows
        last_highs = [s['price'] for s in swing_highs[-3:]]
        last_lows = [s['price'] for s in swing_lows[-3:]]
        
        if (last_highs[-1] > last_highs[-2] and 
            last_lows[-1] > last_lows[-2]):
            return 'bullish'
        elif (last_highs[-1] < last_highs[-2] and 
              last_lows[-1] < last_lows[-2]):
            return 'bearish'
        else:
            return 'ranging'
    
    def detect_order_blocks(self, df: pd.DataFrame, structure: Dict) -> List[OrderBlock]:
        """Detect order blocks from swing points"""
        order_blocks = []
        
        # Look for bullish order blocks (large bearish candles before a move up)
        # and bearish order blocks (large bullish candles before a move down)
        
        for i in range(3, len(df) - 3):
            current_candle = df.iloc[i]
            
            # Bullish order block detection
            if (current_candle['close'] > current_candle['open'] and
                current_candle['high'] - current_candle['low'] > 
                df['high'].rolling(20).mean().iloc[i] * 1.5):
                
                # Check if followed by continuation up
                if (df.iloc[i+1]['close'] > current_candle['high'] and
                    df.iloc[i+2]['close'] > df.iloc[i+1]['close']):
                    
                    ob = OrderBlock(
                        start_price=current_candle['low'],
                        end_price=current_candle['high'],
                        start_time=df.index[i],
                        end_time=df.index[i],
                        type='bullish',
                        strength=self._calculate_ob_strength(df, i),
                        mitigated=False
                    )
                    order_blocks.append(ob)
        
        return order_blocks
    
    def _calculate_ob_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate order block strength based on volume and candle size"""
        candle = df.iloc[index]
        avg_volume = df['volume'].rolling(20).mean().iloc[index]
        avg_range = df['high'].rolling(20).mean().iloc[index] - df['low'].rolling(20).mean().iloc[index]
        
        volume_ratio = candle['volume'] / avg_volume if avg_volume > 0 else 1
        range_ratio = (candle['high'] - candle['low']) / avg_range if avg_range > 0 else 1
        
        strength = (volume_ratio * 0.5 + range_ratio * 0.5)
        return min(strength, 1.0)  # Normalize to [0, 1]
    
    def detect_fvg(self, df: pd.DataFrame) -> List[FVG]:
        """Detect Fair Value Gaps"""
        fvgs = []
        
        for i in range(2, len(df) - 2):
            # Bullish FVG: gap between high of candle i-2 and low of candle i
            if df.iloc[i-2]['high'] > df.iloc[i]['low']:
                fvg = FVG(
                    top=df.iloc[i-2]['high'],
                    bottom=df.iloc[i]['low'],
                    start_time=df.index[i-2],
                    end_time=df.index[i],
                    type='bullish',
                    fill_percentage=self._calculate_fvg_fill(df, i-2, i),
                    premium_zone=self._is_premium_zone(df, df.iloc[i-2]['high'], df.iloc[i]['low'])
                )
                fvgs.append(fvg)
            
            # Bearish FVG: gap between low of candle i-2 and high of candle i
            if df.iloc[i-2]['low'] < df.iloc[i]['high']:
                fvg = FVG(
                    top=df.iloc[i]['high'],
                    bottom=df.iloc[i-2]['low'],
                    start_time=df.index[i-2],
                    end_time=df.index[i],
                    type='bearish',
                    fill_percentage=self._calculate_fvg_fill(df, i-2, i),
                    premium_zone=self._is_premium_zone(df, df.iloc[i]['high'], df.iloc[i-2]['low'])
                )
                fvgs.append(fvg)
        
        return fvgs
    
    def _calculate_fvg_fill(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate how much of the FVG has been filled"""
        # Implementation
        return 0.0
    
    def _is_premium_zone(self, df: pd.DataFrame, top: float, bottom: float) -> bool:
        """Determine if price zone is in premium or discount"""
        current_price = df.iloc[-1]['close']
        range_mid = (top + bottom) / 2
        
        if current_price > range_mid:
            return True  # Premium zone
        else:
            return False  # Discount zone
    
    def detect_liquidity_zones(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """Detect liquidity zones (sweeps and pools)"""
        liquidity_zones = []
        
        # Look for areas where price has swept previous highs/lows
        for i in range(5, len(df) - 5):
            window_highs = df['high'].iloc[i-5:i].max()
            window_lows = df['low'].iloc[i-5:i].min()
            
            # Buy-side liquidity sweep
            if df.iloc[i]['high'] > window_highs and df.iloc[i]['close'] < window_highs:
                liquidity_zones.append(
                    LiquidityZone(
                        price_level=window_highs,
                        type='buy_sweep',
                        strength=self._calculate_liquidity_strength(df, i, window_highs),
                        timestamp=df.index[i]
                    )
                )
            
            # Sell-side liquidity sweep
            if df.iloc[i]['low'] < window_lows and df.iloc[i]['close'] > window_lows:
                liquidity_zones.append(
                    LiquidityZone(
                        price_level=window_lows,
                        type='sell_sweep',
                        strength=self._calculate_liquidity_strength(df, i, window_lows),
                        timestamp=df.index[i]
                    )
                )
        
        return liquidity_zones
    
    def _calculate_liquidity_strength(self, df: pd.DataFrame, index: int, price_level: float) -> float:
        """Calculate liquidity zone strength based on volume and price action"""
        # Implementation
        return 0.7  # Placeholder
    
    def calculate_bos_chochn(self, df: pd.DataFrame, structure: Dict) -> Dict:
        """Calculate Break of Structure and Change of Character"""
        bos_events = []
        choch_events = []
        
        swing_highs = structure['swing_highs']
        swing_lows = structure['swing_lows']
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # BOS - Break of Structure
            last_high = swing_highs[-1]['price']
            prev_high = swing_highs[-2]['price']
            last_low = swing_lows[-1]['price']
            prev_low = swing_lows[-2]['price']
            
            current_close = df.iloc[-1]['close']
            
            # Bullish BOS
            if current_close > last_high and last_high > prev_high:
                bos_events.append({
                    'type': 'bullish',
                    'price': current_close,
                    'timestamp': df.index[-1]
                })
            
            # Bearish BOS
            if current_close < last_low and last_low < prev_low:
                bos_events.append({
                    'type': 'bearish',
                    'price': current_close,
                    'timestamp': df.index[-1]
                })
            
            # CHOCH - Change of Character
            # Detect when trend is changing
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                # Implementation for CHOCH detection
                pass
        
        return {
            'bos_events': bos_events,
            'choch_events': choch_events
        }
    
    def get_all_features(self, df: pd.DataFrame) -> Dict:
        """Generate all SMC features"""
        structure = self.calculate_market_structure(df)
        order_blocks = self.detect_order_blocks(df, structure)
        fvgs = self.detect_fvg(df)
        liquidity_zones = self.detect_liquidity_zones(df)
        bos_choch = self.calculate_bos_chochn(df, structure)
        
        return {
            'structure': structure,
            'order_blocks': order_blocks,
            'fvgs': fvgs,
            'liquidity_zones': liquidity_zones,
            'bos_choch': bos_choch
        }