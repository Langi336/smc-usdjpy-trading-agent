# src/signal_generation/signal_generator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    direction: str
    entry_zone: Tuple[float, float]
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    rr_ratio: float
    confidence: float
    timeframe: str
    timestamp: datetime
    rationale: List[str]
    risk_percentage: float
    htf_bias: str
    ltf_signal: str

class SignalGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.min_confidence = config.get('min_confidence', 0.6)
        self.max_risk = config.get('max_risk', 0.02)
        self.min_rr_ratio = config.get('min_rr_ratio', 1.5)
        self.multi_tf = config.get('multi_timeframe', {})
        
        # Initialize multi-timeframe analyzer
        from src.smc_engine.multi_timeframe import MultiTimeframeAnalyzer
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.multi_tf)
        
    def generate_signal(self, data: Dict[str, pd.DataFrame], 
                        smc_data: Dict, 
                        model_predictions: Dict) -> Optional[TradingSignal]:
        """Generate trading signal using multi-timeframe analysis"""
        
        # 1. Run multi-timeframe analysis
        mtf_result = self.mtf_analyzer.analyze(data)
        
        logger.info(f"MTF Analysis: Bias={mtf_result['htf_analysis']['bias']}, "
                   f"Score={mtf_result['confluence']['score']:.2%}")
        
        # 2. Check if we have a valid signal
        if mtf_result['signal'] is None:
            logger.info("No valid MTF signal detected")
            return None
        
        signal = mtf_result['signal']
        
        # 3. Check confidence
        if signal['confidence'] < self.min_confidence:
            logger.info(f"Low confidence: {signal['confidence']:.2%}")
            return None
        
        # 4. Get entry zones from MTF analysis
        entry_zones = mtf_result.get('entry_zones', [])
        if not entry_zones:
            logger.info("No entry zones found")
            return None
        
        # 5. Build the final trading signal
        return self._build_trading_signal(
            signal, 
            entry_zones, 
            mtf_result,
            data
        )
    
    def _build_trading_signal(self, signal: Dict, 
                              entry_zones: List[Dict],
                              mtf_result: Dict,
                              data: Dict[str, pd.DataFrame]) -> TradingSignal:
        """Build the final trading signal"""
        
        # Get the best entry zone
        best_entry = entry_zones[0]
        entry_zone = best_entry['zone']
        
        # Calculate stop loss
        stop_loss = self._calculate_stop_loss_mtf(entry_zone, signal['direction'], data)
        
        # Calculate take profits
        take_profits = self._calculate_take_profits_mtf(entry_zone, stop_loss, signal['direction'])
        
        # Calculate RR ratio
        entry_price = (entry_zone[0] + entry_zone[1]) / 2
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profits[1] - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Build rationale
        rationale = [
            f"HTF Bias: {mtf_result['htf_analysis']['bias']}",
            f"LTF Signal: {signal['direction']}",
            f"Confluence Score: {mtf_result['confluence']['score']:.2%}",
            f"Entry Zone: {entry_zone[0]:.4f} - {entry_zone[1]:.4f}"
        ]
        
        if mtf_result['confluence']['details']:
            rationale.extend(mtf_result['confluence']['details'])
        
        return TradingSignal(
            direction=signal['direction'],
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit_1=take_profits[0],
            take_profit_2=take_profits[1],
            take_profit_3=take_profits[2],
            rr_ratio=rr_ratio,
            confidence=signal['confidence'],
            timeframe="1H-2H bias, 5M-15M entry",
            timestamp=datetime.now(),
            rationale=rationale,
            risk_percentage=self.max_risk,
            htf_bias=mtf_result['htf_analysis']['bias'],
            ltf_signal=signal['direction']
        )
    
    def _calculate_stop_loss_mtf(self, entry_zone: Tuple[float, float], 
                                 direction: str,
                                 data: Dict[str, pd.DataFrame]) -> float:
        """Calculate stop loss using multi-timeframe levels"""
        
        entry_price = (entry_zone[0] + entry_zone[1]) / 2
        
        if direction == 'long':
            # Look for support on LTF
            for tf in ['15M', '5M']:
                if tf in data and not data[tf].empty:
                    df = data[tf]
                    recent_lows = df['low'].rolling(20).min().iloc[-10:]
                    support = min(recent_lows)
                    if support < entry_price:
                        # Place SL slightly below support
                        return support * 0.998
            # Default SL
            return entry_price * 0.99
            
        elif direction == 'short':
            # Look for resistance on LTF
            for tf in ['15M', '5M']:
                if tf in data and not data[tf].empty:
                    df = data[tf]
                    recent_highs = df['high'].rolling(20).max().iloc[-10:]
                    resistance = max(recent_highs)
                    if resistance > entry_price:
                        # Place SL slightly above resistance
                        return resistance * 1.002
            # Default SL
            return entry_price * 1.01
        
        return entry_price
    
    def _calculate_take_profits_mtf(self, entry_zone: Tuple[float, float],
                                    stop_loss: float,
                                    direction: str) -> Tuple[float, float, float]:
        """Calculate take profits using MTF levels"""
        
        entry_price = (entry_zone[0] + entry_zone[1]) / 2
        risk = abs(entry_price - stop_loss)
        
        if direction == 'long':
            tp1 = entry_price + risk * 1.5
            tp2 = entry_price + risk * 3.0
            tp3 = entry_price + risk * 5.0
        else:
            tp1 = entry_price - risk * 1.5
            tp2 = entry_price - risk * 3.0
            tp3 = entry_price - risk * 5.0
        
        return (tp1, tp2, tp3)