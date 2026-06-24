# src/smc_engine/multi_timeframe.py
"""
Multi-Timeframe Analysis Engine
Analyzes higher timeframes (1H-2H) for bias and lower timeframes (5M-15M) for entries
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.htf_timeframes = config.get('htf_timeframes', ['1H', '2H'])
        self.ltf_timeframes = config.get('ltf_timeframes', ['5M', '15M'])
        self.min_confidence = config.get('min_confidence', 0.6)
        
    def analyze(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze multi-timeframe data
        data: {'1H': df_1h, '2H': df_2h, '5M': df_5m, '15M': df_15m}
        """
        result = {
            'htf_analysis': {},
            'ltf_analysis': {},
            'confluence': {},
            'signal': None,
            'entry_zones': [],
            'timestamp': datetime.now()
        }
        
        # 1. Analyze Higher Timeframes (1H-2H)
        htf_result = self._analyze_htf(data)
        result['htf_analysis'] = htf_result
        
        # 2. Analyze Lower Timeframes (5M-15M)
        ltf_result = self._analyze_ltf(data)
        result['ltf_analysis'] = ltf_result
        
        # 3. Check for confluence between HTF and LTF
        confluence = self._check_confluence(htf_result, ltf_result)
        result['confluence'] = confluence
        
        # 4. Generate signal if confluence is strong
        if confluence['score'] >= self.min_confidence:
            signal = self._generate_signal(htf_result, ltf_result, confluence)
            result['signal'] = signal
            
            # 5. Calculate entry zones on LTF
            entry_zones = self._calculate_entry_zones(data, htf_result, ltf_result)
            result['entry_zones'] = entry_zones
        
        return result
    
    def _analyze_htf(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze higher timeframes for overall bias"""
        htf_analysis = {
            'bias': 'neutral',
            'trend': 'ranging',
            'key_levels': {'support': [], 'resistance': []},
            'order_blocks': [],
            'fvgs': [],
            'structure': {},
            'confidence': 0.0
        }
        
        # Analyze each HTF
        htf_results = []
        for tf in self.htf_timeframes:
            if tf in data and not data[tf].empty:
                tf_result = self._analyze_single_timeframe(data[tf], tf)
                htf_results.append(tf_result)
        
        if not htf_results:
            return htf_analysis
        
        # Combine results from multiple HTFs
        # Count bullish vs bearish signals
        bullish_count = sum(1 for r in htf_results if r['bias'] == 'bullish')
        bearish_count = sum(1 for r in htf_results if r['bias'] == 'bearish')
        
        if bullish_count > bearish_count:
            htf_analysis['bias'] = 'bullish'
            htf_analysis['confidence'] = bullish_count / len(htf_results)
        elif bearish_count > bullish_count:
            htf_analysis['bias'] = 'bearish'
            htf_analysis['confidence'] = bearish_count / len(htf_results)
        
        # Aggregate trends
        trends = [r['trend'] for r in htf_results if r['trend'] != 'ranging']
        if trends:
            htf_analysis['trend'] = max(set(trends), key=trends.count)
        
        # Aggregate key levels
        for r in htf_results:
            htf_analysis['key_levels']['support'].extend(r.get('key_levels', {}).get('support', []))
            htf_analysis['key_levels']['resistance'].extend(r.get('key_levels', {}).get('resistance', []))
            htf_analysis['order_blocks'].extend(r.get('order_blocks', []))
            htf_analysis['fvgs'].extend(r.get('fvgs', []))
        
        # Get the most recent structure
        if htf_results:
            htf_analysis['structure'] = htf_results[-1].get('structure', {})
        
        return htf_analysis
    
    def _analyze_ltf(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze lower timeframes for entry opportunities"""
        ltf_analysis = {
            'bias': 'neutral',
            'entry_signals': [],
            'price_action': {},
            'momentum': 0.0,
            'volume_profile': {},
            'confidence': 0.0
        }
        
        # Analyze each LTF
        ltf_results = []
        for tf in self.ltf_timeframes:
            if tf in data and not data[tf].empty:
                tf_result = self._analyze_single_timeframe(data[tf], tf)
                ltf_results.append(tf_result)
        
        if not ltf_results:
            return ltf_analysis
        
        # Check for entry signals on LTF
        for r in ltf_results:
            if r.get('entry_signal'):
                ltf_analysis['entry_signals'].append({
                    'timeframe': r['timeframe'],
                    'signal': r['entry_signal'],
                    'price': r.get('current_price', 0),
                    'strength': r.get('signal_strength', 0.5)
                })
        
        # Calculate momentum
        momentum_values = [r.get('momentum', 0) for r in ltf_results]
        if momentum_values:
            ltf_analysis['momentum'] = sum(momentum_values) / len(momentum_values)
        
        # Determine LTF bias
        if ltf_analysis['entry_signals']:
            bullish_signals = sum(1 for s in ltf_analysis['entry_signals'] 
                                if s['signal'] == 'buy')
            bearish_signals = sum(1 for s in ltf_analysis['entry_signals'] 
                                if s['signal'] == 'sell')
            
            if bullish_signals > bearish_signals:
                ltf_analysis['bias'] = 'bullish'
                ltf_analysis['confidence'] = bullish_signals / len(ltf_analysis['entry_signals'])
            elif bearish_signals > bullish_signals:
                ltf_analysis['bias'] = 'bearish'
                ltf_analysis['confidence'] = bearish_signals / len(ltf_analysis['entry_signals'])
        
        return ltf_analysis
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze a single timeframe"""
        result = {
            'timeframe': timeframe,
            'bias': 'neutral',
            'trend': 'ranging',
            'key_levels': {'support': [], 'resistance': []},
            'order_blocks': [],
            'fvgs': [],
            'structure': {},
            'entry_signal': None,
            'signal_strength': 0.0,
            'momentum': 0.0,
            'current_price': df.iloc[-1]['close'] if not df.empty else 0
        }
        
        if df.empty:
            return result
        
        # Calculate basic indicators
        df = df.copy()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        current_price = df.iloc[-1]['close']
        current_rsi = df.iloc[-1]['rsi']
        
        # Determine trend
        if current_price > df.iloc[-1]['sma_50'] and df.iloc[-1]['sma_20'] > df.iloc[-1]['sma_50']:
            result['trend'] = 'bullish'
        elif current_price < df.iloc[-1]['sma_50'] and df.iloc[-1]['sma_20'] < df.iloc[-1]['sma_50']:
            result['trend'] = 'bearish'
        
        # Determine bias based on multiple factors
        bias_score = 0
        
        # RSI
        if current_rsi < 30:
            bias_score += 1  # Oversold - bullish
        elif current_rsi > 70:
            bias_score -= 1  # Overbought - bearish
        
        # MACD
        if df.iloc[-1]['macd'] > df.iloc[-1]['macd_signal']:
            bias_score += 1
        else:
            bias_score -= 1
        
        # Price vs SMA
        if current_price > df.iloc[-1]['sma_20']:
            bias_score += 1
        else:
            bias_score -= 1
        
        # Determine final bias
        if bias_score >= 2:
            result['bias'] = 'bullish'
        elif bias_score <= -2:
            result['bias'] = 'bearish'
        
        # Calculate momentum
        result['momentum'] = (df.iloc[-1]['close'] - df.iloc[-10]['close']) / df.iloc[-10]['close'] * 100
        
        # Detect entry signals
        entry_signal, strength = self._detect_entry_signal(df, timeframe)
        result['entry_signal'] = entry_signal
        result['signal_strength'] = strength
        
        # Find key levels
        result['key_levels'] = self._find_key_levels(df)
        
        # Detect order blocks (simplified)
        result['order_blocks'] = self._detect_order_blocks_simple(df)
        
        # Detect FVGs (simplified)
        result['fvgs'] = self._detect_fvgs_simple(df)
        
        return result
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def _detect_entry_signal(self, df: pd.DataFrame, timeframe: str) -> Tuple[Optional[str], float]:
        """Detect entry signals on a timeframe"""
        if len(df) < 20:
            return None, 0.0
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Simple entry signals based on price action
        signal = None
        strength = 0.0
        
        # Bullish engulfing
        if (last_candle['close'] > last_candle['open'] and
            prev_candle['close'] < prev_candle['open'] and
            last_candle['close'] > prev_candle['open'] and
            last_candle['open'] < prev_candle['close']):
            signal = 'buy'
            strength = 0.7
        
        # Bearish engulfing
        elif (last_candle['close'] < last_candle['open'] and
              prev_candle['close'] > prev_candle['open'] and
              last_candle['close'] < prev_candle['open'] and
              last_candle['open'] > prev_candle['close']):
            signal = 'sell'
            strength = 0.7
        
        # Breakout above resistance
        elif last_candle['close'] > df['sma_20'].iloc[-1] * 1.02:
            signal = 'buy'
            strength = 0.6
        
        # Breakout below support
        elif last_candle['close'] < df['sma_20'].iloc[-1] * 0.98:
            signal = 'sell'
            strength = 0.6
        
        return signal, strength
    
    def _find_key_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Find key support and resistance levels"""
        levels = {'support': [], 'resistance': []}
        
        if len(df) < 30:
            return levels
        
        # Find swing highs and lows
        window = 5
        highs = df['high'].values
        lows = df['low'].values
        
        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                levels['resistance'].append(highs[i])
            if lows[i] == min(lows[i-window:i+window+1]):
                levels['support'].append(lows[i])
        
        # Remove duplicates and sort
        levels['support'] = sorted(set(levels['support']), reverse=True)[:3]
        levels['resistance'] = sorted(set(levels['resistance']))[:3]
        
        return levels
    
    def _detect_order_blocks_simple(self, df: pd.DataFrame) -> List[Dict]:
        """Simple order block detection"""
        order_blocks = []
        
        for i in range(3, len(df) - 2):
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            # Bullish order block
            if (candle['close'] < candle['open'] and  # Red candle
                prev_candle['close'] < prev_candle['open'] and  # Previous red
                df.iloc[i+1]['close'] > candle['high']):  # Next candle moves up
                order_blocks.append({
                    'type': 'bullish',
                    'high': candle['high'],
                    'low': candle['low'],
                    'time': candle.name,
                    'strength': abs(candle['close'] - candle['open']) / candle['open']
                })
            
            # Bearish order block
            elif (candle['close'] > candle['open'] and  # Green candle
                  prev_candle['close'] > prev_candle['open'] and  # Previous green
                  df.iloc[i+1]['close'] < candle['low']):  # Next candle moves down
                order_blocks.append({
                    'type': 'bearish',
                    'high': candle['high'],
                    'low': candle['low'],
                    'time': candle.name,
                    'strength': abs(candle['close'] - candle['open']) / candle['open']
                })
        
        return order_blocks[:5]  # Return most recent 5
    
    def _detect_fvgs_simple(self, df: pd.DataFrame) -> List[Dict]:
        """Simple FVG detection"""
        fvgs = []
        
        for i in range(2, len(df) - 1):
            # Bullish FVG
            if df.iloc[i-2]['high'] > df.iloc[i]['low']:
                fvgs.append({
                    'type': 'bullish',
                    'top': df.iloc[i-2]['high'],
                    'bottom': df.iloc[i]['low'],
                    'time': df.index[i],
                    'size': df.iloc[i-2]['high'] - df.iloc[i]['low']
                })
            
            # Bearish FVG
            if df.iloc[i-2]['low'] < df.iloc[i]['high']:
                fvgs.append({
                    'type': 'bearish',
                    'top': df.iloc[i]['high'],
                    'bottom': df.iloc[i-2]['low'],
                    'time': df.index[i],
                    'size': df.iloc[i]['high'] - df.iloc[i-2]['low']
                })
        
        return fvgs[:5]  # Return most recent 5
    
    def _check_confluence(self, htf: Dict, ltf: Dict) -> Dict:
        """Check for confluence between HTF and LTF"""
        confluence = {
            'score': 0.0,
            'bias_alignment': False,
            'level_confluence': False,
            'signal_confirmation': False,
            'details': []
        }
        
        score = 0.0
        
        # Check bias alignment
        if htf['bias'] != 'neutral' and ltf['bias'] != 'neutral':
            if htf['bias'] == ltf['bias']:
                confluence['bias_alignment'] = True
                score += 0.4
                confluence['details'].append(f"Bias aligned: {htf['bias']}")
        
        # Check if price is at key levels
        current_price = ltf.get('current_price', 0)
        for level in htf.get('key_levels', {}).get('support', []):
            if abs(current_price - level) / current_price < 0.01:
                confluence['level_confluence'] = True
                score += 0.3
                confluence['details'].append(f"At support level: {level:.4f}")
                break
        
        for level in htf.get('key_levels', {}).get('resistance', []):
            if abs(current_price - level) / current_price < 0.01:
                confluence['level_confluence'] = True
                score += 0.3
                confluence['details'].append(f"At resistance level: {level:.4f}")
                break
        
        # Check for entry signals on LTF
        if ltf.get('entry_signals'):
            confluence['signal_confirmation'] = True
            score += 0.3
            confluence['details'].append(f"Entry signals on LTF: {len(ltf['entry_signals'])}")
        
        confluence['score'] = min(score, 1.0)
        
        return confluence
    
    def _generate_signal(self, htf: Dict, ltf: Dict, confluence: Dict) -> Dict:
        """Generate final trading signal"""
        signal = {
            'direction': 'neutral',
            'confidence': confluence['score'],
            'entry_zones': [],
            'stop_loss': None,
            'take_profit': None,
            'timeframe': 'Mixed',
            'rationale': confluence['details']
        }
        
        # Determine direction based on HTF bias and LTF signals
        if htf['bias'] == 'bullish' and ltf['bias'] == 'bullish':
            signal['direction'] = 'long'
        elif htf['bias'] == 'bearish' and ltf['bias'] == 'bearish':
            signal['direction'] = 'short'
        elif htf['bias'] == 'bullish' and ltf['entry_signals']:
            # HTF bullish, LTF has entry signals - look for buy signals
            buy_signals = [s for s in ltf['entry_signals'] if s['signal'] == 'buy']
            if buy_signals:
                signal['direction'] = 'long'
        elif htf['bias'] == 'bearish' and ltf['entry_signals']:
            # HTF bearish, LTF has entry signals - look for sell signals
            sell_signals = [s for s in ltf['entry_signals'] if s['signal'] == 'sell']
            if sell_signals:
                signal['direction'] = 'short'
        
        return signal
    
    def _calculate_entry_zones(self, data: Dict[str, pd.DataFrame], 
                               htf: Dict, ltf: Dict) -> List[Dict]:
        """Calculate precise entry zones on LTF"""
        entry_zones = []
        
        # Get the lowest timeframe data
        ltf_data = None
        for tf in self.ltf_timeframes:
            if tf in data and not data[tf].empty:
                ltf_data = data[tf]
                break
        
        if ltf_data is None:
            return entry_zones
        
        current_price = ltf_data.iloc[-1]['close']
        
        # Calculate entry zones
        if htf['bias'] == 'bullish':
            # Look for support levels on LTF
            supports = ltf_data['low'].rolling(20).min().iloc[-5:].values
            
            if len(supports) >= 3:
                nearest_support = max([s for s in supports if s < current_price])
                entry_zones.append({
                    'type': 'buy',
                    'zone': (nearest_support, nearest_support * 1.002),
                    'strength': 'strong'
                })
        
        elif htf['bias'] == 'bearish':
            # Look for resistance levels on LTF
            resistances = ltf_data['high'].rolling(20).max().iloc[-5:].values
            
            if len(resistances) >= 3:
                nearest_resistance = min([r for r in resistances if r > current_price])
                entry_zones.append({
                    'type': 'sell',
                    'zone': (nearest_resistance * 0.998, nearest_resistance),
                    'strength': 'strong'
                })
        
        return entry_zones