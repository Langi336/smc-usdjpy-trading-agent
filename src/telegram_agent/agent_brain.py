# src/telegram_agent/agent_brain.py
import openai
from typing import Dict, List, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TelegramAgentBrain:
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config['openai_api_key']
        openai.api_key = self.api_key
        self.model = config.get('model', 'gpt-4o')
        self.smc_knowledge = self._load_smc_knowledge()
        
    def _load_smc_knowledge(self):
        """Load SMC knowledge base"""
        return {
            'order_blocks': {
                'definition': 'Order blocks are large candles that contain institutional orders...',
                'bullish': 'A bullish order block is a strong down candle followed by a move up...',
                'bearish': 'A bearish order block is a strong up candle followed by a move down...'
            },
            'fvg': {
                'definition': 'Fair Value Gaps occur when there is an imbalance in price...',
                'types': 'Bullish FVG is formed when price gaps up, bearish when it gaps down...',
                'trading': 'Price often returns to fill FVGs before continuing trend...'
            },
            'liquidity': {
                'definition': 'Liquidity zones are areas where stop losses and pending orders are concentrated...',
                'sweeps': 'Liquidity sweeps occur when price breaks above highs or below lows...',
                'trading': 'Price often moves to liquidity before reversing...'
            }
        }
    
    def generate_signal_narrative(self, signal: Dict, smc_data: Dict) -> str:
        """Generate human-readable signal explanation"""
        prompt = self._build_signal_prompt(signal, smc_data)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional forex trading analyst specializing in Smart Money Concepts. Provide clear, concise trade analysis with reasoning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return self._fallback_narrative(signal)
    
    def _build_signal_prompt(self, signal: Dict, smc_data: Dict) -> str:
        """Build prompt for signal analysis"""
        prompt = f"""
        Analyze this USD/JPY trading signal based on Smart Money Concepts:
        
        Direction: {signal.get('direction', 'N/A')}
        Entry Zone: {signal.get('entry_zone', 'N/A')}
        Stop Loss: {signal.get('stop_loss', 'N/A')}
        Take Profit Levels: {signal.get('take_profit_1', 'N/A')}, {signal.get('take_profit_2', 'N/A')}, {signal.get('take_profit_3', 'N/A')}
        Risk/Reward: {signal.get('rr_ratio', 'N/A')}
        Confidence: {signal.get('confidence', 'N/A')}
        
        SMC Analysis:
        - Market Structure: {smc_data.get('structure', {}).get('trend', 'N/A')}
        - Order Blocks: {len(smc_data.get('order_blocks', []))} detected
        - FVGs: {len(smc_data.get('fvgs', []))} detected
        - Liquidity Zones: {len(smc_data.get('liquidity_zones', []))} detected
        
        Provide a clear analysis explaining:
        1. Why this trade setup is valid
        2. Key SMC concepts at play
        3. Risk management considerations
        4. What to watch for
        """
        return prompt
    
    def _fallback_narrative(self, signal: Dict) -> str:
        """Generate fallback narrative if API fails"""
        direction = signal.get('direction', '').upper()
        entry = signal.get('entry_zone', (0, 0))
        sl = signal.get('stop_loss', 0)
        
        return f"""
        🎯 {direction} SIGNAL DETECTED
        
        Entry Zone: {entry[0]:.4f} - {entry[1]:.4f}
        Stop Loss: {sl:.4f}
        
        Based on Smart Money Concepts analysis, we identify a potential {direction} trade opportunity.
        The setup shows confluence between market structure and order flow.
        
        Please view the attached chart for detailed technical analysis.
        """
    
    def answer_question(self, question: str, context: Dict) -> str:
        """Answer user questions using RAG"""
        # Search SMC knowledge base
        relevant_knowledge = self._search_knowledge(question)
        
        # Build prompt
        prompt = f"""
        Context: {json.dumps(context, indent=2)}
        Knowledge: {json.dumps(relevant_knowledge, indent=2)}
        
        User Question: {question}
        
        Provide a clear, accurate answer based on the context and your knowledge of Smart Money Concepts.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a forex trading expert specializing in Smart Money Concepts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "I'm having trouble processing your question. Please try again later."
    
    def _search_knowledge(self, query: str) -> Dict:
        """Search SMC knowledge base for relevant information"""
        results = {}
        query_lower = query.lower()
        
        for topic, content in self.smc_knowledge.items():
            if any(keyword in query_lower for keyword in [topic, topic.replace('_', ' ')]):
                results[topic] = content
        
        return results
    
    def generate_performance_report(self, stats: Dict) -> str:
        """Generate performance report"""
        win_rate = stats.get('win_rate', 0) * 100
        avg_rr = stats.get('avg_rr', 0)
        total_trades = stats.get('total_trades', 0)
        profit = stats.get('profit', 0)
        sharpe = stats.get('sharpe', 0)
        max_dd = stats.get('max_drawdown', 0) * 100
        
        return f"""
        📊 PERFORMANCE REPORT
        
        Total Trades: {total_trades}
        Win Rate: {win_rate:.1f}%
        Average RR: {avg_rr:.2f}
        Total P&L: ${profit:,.2f}
        Sharpe Ratio: {sharpe:.2f}
        Max Drawdown: {max_dd:.1f}%
        
        💡 Analysis: 
        {self._generate_performance_insights(stats)}
        """
    
    def _generate_performance_insights(self, stats: Dict) -> str:
        """Generate insights from performance stats"""
        insights = []
        
        if stats.get('win_rate', 0) > 0.55:
            insights.append("✅ Strong win rate above 55%")
        elif stats.get('win_rate', 0) > 0.45:
            insights.append("⚠️ Win rate is acceptable but room for improvement")
        else:
            insights.append("❌ Win rate below 45% - consider reviewing strategy")
        
        if stats.get('sharpe', 0) > 1.5:
            insights.append("✅ Excellent risk-adjusted returns")
        elif stats.get('sharpe', 0) > 0.8:
            insights.append("⚠️ Acceptable risk-adjusted returns")
        else:
            insights.append("❌ Low Sharpe ratio - review risk management")
        
        return "\n".join(insights)