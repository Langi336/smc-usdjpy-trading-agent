# main.py - Updated with Multi-Timeframe Support
from typing import Dict, List, Tuple, Optional
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class USDJPYAgent:
    def __init__(self):
        print("🚀 Initializing USD/JPY SMC Agent with Multi-Timeframe Analysis...")
        
        # Load config
        try:
            with open('config/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
            print("✅ Config loaded")
        except:
            print("⚠️ Using default config")
            self.config = {
                'symbol': 'USD/JPY',
                'multi_timeframe': {
                    'htf_timeframes': ['1H', '2H'],
                    'ltf_timeframes': ['5M', '15M'],
                    'min_confidence': 0.6
                }
            }
        
        # Initialize components
        try:
            from src.signal_generation.signal_generator import SignalGenerator
            self.signal_generator = SignalGenerator(self.config)
            print("✅ Signal generator initialized")
        except Exception as e:
            print(f"⚠️ Signal generator error: {e}")
            self.signal_generator = None
        
        self.running = True
        self.trade_count = 0
        self.trades = []
        
        print("✅ Agent ready!")
        print(f"📊 Timeframes: HTF={self.config.get('multi_timeframe', {}).get('htf_timeframes', ['1H', '2H'])}")
        print(f"📊 Entry Timeframes: LTF={self.config.get('multi_timeframe', {}).get('ltf_timeframes', ['5M', '15M'])}")
        print("\n⏰ Analyzing 1H-2H for bias")
        print("🎯 Taking entries on 5M-15M\n")
    
    def generate_mock_data(self) -> Dict[str, pd.DataFrame]:
        """Generate mock data for testing all timeframes"""
        data = {}
        
        end_time = datetime.now()
        
        # Generate data for each timeframe
        timeframes = ['1H', '2H', '5M', '15M']
        
        for tf in timeframes:
            # Calculate number of candles based on timeframe
            if tf == '1H':
                periods = 24 * 3  # 3 days of 1H data
                freq = 'H'
            elif tf == '2H':
                periods = 12 * 3  # 3 days of 2H data
                freq = '2H'
            elif tf == '15M':
                periods = 4 * 24 * 3  # 3 days of 15M data
                freq = '15T'
            else:  # 5M
                periods = 12 * 24 * 3  # 3 days of 5M data
                freq = '5T'
            
            # Generate trend data
            base_price = 150
            trend = np.linspace(0, np.random.uniform(-2, 2), periods)
            noise = np.random.normal(0, 0.5, periods)
            prices = base_price + trend + noise
            
            # Create OHLCV data
            dates = pd.date_range(end=end_time, periods=periods, freq=freq)
            df = pd.DataFrame({
                'open': prices[:-1] if len(prices) > 1 else prices,
                'high': prices + np.abs(np.random.normal(0, 0.3, periods)),
                'low': prices - np.abs(np.random.normal(0, 0.3, periods)),
                'close': prices,
                'volume': np.random.normal(1000, 200, periods)
            }, index=dates[-periods:])
            
            # Ensure correct length
            if len(df) > periods:
                df = df.iloc[-periods:]
            
            data[tf] = df
        
        return data
    
    async def analyze_market(self) -> Dict:
        """Analyze market using multi-timeframe analysis"""
        if self.signal_generator:
            try:
                # Generate mock data for all timeframes
                data = self.generate_mock_data()
                
                # Generate signal using multi-timeframe analysis
                signal = self.signal_generator.generate_signal(data, {}, {})
                
                if signal:
                    return {
                        'signal': signal,
                        'htf_bias': signal.htf_bias,
                        'ltf_signal': signal.ltf_signal,
                        'confidence': signal.confidence,
                        'entry_zone': signal.entry_zone,
                        'stop_loss': signal.stop_loss,
                        'take_profits': [signal.take_profit_1, signal.take_profit_2, signal.take_profit_3],
                        'rr_ratio': signal.rr_ratio,
                        'rationale': signal.rationale,
                        'direction': signal.direction
                    }
                else:
                    return {'signal': None}
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return {'signal': None, 'error': str(e)}
        
        # Fallback
        return {
            'signal': None,
            'direction': 'neutral',
            'confidence': 0.5
        }
    
    async def generate_chart(self, signal_data: Dict) -> Optional[str]:
        """Generate chart with multi-timeframe analysis"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            if not signal_data or not signal_data.get('signal'):
                return None
            
            signal = signal_data['signal']
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
            
            # Main chart - HTF bias
            ax1 = plt.subplot(gs[0, :])
            ax1.set_title(f"USD/JPY - HTF Bias: {signal.htf_bias.upper()}")
            
            # Plot price
            data = self.generate_mock_data()
            if '1H' in data:
                df = data['1H']
                ax1.plot(df.index, df['close'], label='1H Price', linewidth=1)
                
                # Add support/resistance
                support = df['low'].rolling(20).min()
                resistance = df['high'].rolling(20).max()
                ax1.plot(df.index, support, 'g--', alpha=0.5, label='Support')
                ax1.plot(df.index, resistance, 'r--', alpha=0.5, label='Resistance')
                
                # Mark entry zone
                entry_zone = signal.entry_zone
                ax1.axhspan(entry_zone[0], entry_zone[1], alpha=0.3, color='yellow', label='Entry Zone')
                
                # Mark SL and TP
                ax1.axhline(y=signal.stop_loss, color='red', linestyle='--', label='Stop Loss')
                ax1.axhline(y=signal.take_profit_1, color='green', linestyle='--', alpha=0.5, label='TP1')
                ax1.axhline(y=signal.take_profit_2, color='blue', linestyle='--', alpha=0.5, label='TP2')
                ax1.axhline(y=signal.take_profit_3, color='purple', linestyle='--', alpha=0.5, label='TP3')
                
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # LTF Entry Signals
            ax2 = plt.subplot(gs[1, :])
            ax2.set_title(f"LTF Entry Signals - {signal.ltf_signal.upper()}")
            
            if '15M' in data:
                df = data['15M']
                ax2.plot(df.index, df['close'], label='15M Price', linewidth=0.5)
                
                # Mark entry signals
                entry_price = (entry_zone[0] + entry_zone[1]) / 2
                ax2.axhline(y=entry_price, color='orange', linestyle='-', label='Entry Price')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Confidence and RR
            ax3 = plt.subplot(gs[2, 0])
            ax3.axis('off')
            ax3.text(0.1, 0.6, f"Confidence: {signal.confidence:.0%}", fontsize=12, fontweight='bold')
            ax3.text(0.1, 0.4, f"RR Ratio: {signal.rr_ratio:.1f}:1", fontsize=12, fontweight='bold')
            ax3.text(0.1, 0.2, f"Direction: {signal.direction.upper()}", fontsize=12, fontweight='bold')
            
            # Rationale
            ax4 = plt.subplot(gs[2, 1])
            ax4.axis('off')
            ax4.text(0.1, 0.8, "📝 Rationale:", fontsize=11, fontweight='bold')
            for i, reason in enumerate(signal.rationale[:4]):
                ax4.text(0.1, 0.6 - i*0.15, f"• {reason}", fontsize=9)
            
            plt.tight_layout()
            
            # Save
            filename = f"chart_mtf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Chart saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None
    
    async def send_telegram(self, signal_data: Dict, chart_file: Optional[str]):
        """Send signal to Telegram"""
        try:
            from telegram import Bot
            
            token = self.config.get('telegram', {}).get('bot_token')
            chat_id = self.config.get('telegram', {}).get('subscribers', [None])[0]
            
            if not token or token == "YOUR_BOT_TOKEN_HERE":
                print("⚠️ Telegram not configured - printing to console instead")
                self._print_signal(signal_data)
                return
            
            bot = Bot(token=token)
            signal = signal_data['signal']
            
            # Build message
            direction_emoji = "🟢" if signal.direction == 'long' else "🔴" if signal.direction == 'short' else "⚪"
            entry = signal.entry_zone
            
            message = f"""
{direction_emoji} *MULTI-TIMEFRAME TRADING SIGNAL*

*Direction:* {signal.direction.upper()}
*HTF Bias:* {signal.htf_bias.upper()} (1H-2H)
*LTF Signal:* {signal.ltf_signal.upper()} (5M-15M)

*Entry Zone:* {entry[0]:.4f} - {entry[1]:.4f}
*Stop Loss:* {signal.stop_loss:.4f}
*Take Profits:*
• TP1: {signal.take_profit_1:.4f}
• TP2: {signal.take_profit_2:.4f}
• TP3: {signal.take_profit_3:.4f}

*Risk/Reward:* {signal.rr_ratio:.1f}:1
*Confidence:* {signal.confidence:.0%}

*📝 Rationale:*
{chr(10).join(['• ' + r for r in signal.rationale])}

⚠️ *Risk Management:* Max 1-2% per trade
"""
            
            await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
            
            if chart_file:
                with open(chart_file, 'rb') as f:
                    await bot.send_photo(chat_id=chat_id, photo=f)
            
            print("✅ Signal sent to Telegram")
            
        except Exception as e:
            print(f"⚠️ Could not send to Telegram: {e}")
            self._print_signal(signal_data)
    
    def _print_signal(self, signal_data: Dict):
        """Print signal to console when Telegram is not available"""
        if not signal_data or not signal_data.get('signal'):
            return
        
        signal = signal_data['signal']
        print("\n" + "="*60)
        print(f"📊 MULTI-TIMEFRAME SIGNAL")
        print("="*60)
        print(f"Direction: {signal.direction.upper()}")
        print(f"HTF Bias: {signal.htf_bias.upper()} (1H-2H)")
        print(f"LTF Signal: {signal.ltf_signal.upper()} (5M-15M)")
        print(f"Entry Zone: {signal.entry_zone[0]:.4f} - {signal.entry_zone[1]:.4f}")
        print(f"Stop Loss: {signal.stop_loss:.4f}")
        print(f"TP1: {signal.take_profit_1:.4f} | TP2: {signal.take_profit_2:.4f} | TP3: {signal.take_profit_3:.4f}")
        print(f"RR Ratio: {signal.rr_ratio:.1f}:1")
        print(f"Confidence: {signal.confidence:.0%}")
        print("\nRationale:")
        for r in signal.rationale:
            print(f"  • {r}")
        print("="*60 + "\n")
    
    async def run(self):
        """Main loop"""
        print("\n📊 Starting multi-timeframe analysis...")
        print("⏰ Analyzing 1H-2H for bias")
        print("🎯 Looking for entries on 5M-15M")
        print("Press Ctrl+C to stop\n")
        
        while self.running:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            self.trade_count += 1
            
            # Get analysis
            result = await self.analyze_market()
            
            if result and result.get('signal'):
                print(f"\n📈 Trade #{self.trade_count}: {result['direction'].upper()} | "
                      f"Confidence: {result['confidence']:.0%} | "
                      f"HTF: {result.get('htf_bias', 'N/A')} | "
                      f"LTF: {result.get('ltf_signal', 'N/A')}")
                
                # Generate chart
                chart_file = await self.generate_chart(result)
                
                # Send to Telegram
                await self.send_telegram(result, chart_file)
                
                # Store trade
                self.trades.append({
                    'time': datetime.now(),
                    'direction': result['direction'],
                    'confidence': result['confidence'],
                    'entry': result.get('entry_zone', (0, 0)),
                    'sl': result.get('stop_loss', 0),
                    'tp': result.get('take_profits', [0, 0, 0]),
                    'rr': result.get('rr_ratio', 0)
                })
            else:
                print(f"⏳ Analysis #{self.trade_count}: No signal | "
                      f"HTF Bias: {result.get('htf_bias', 'N/A') if result else 'N/A'}")

async def main():
    """Main entry point"""
    agent = USDJPYAgent()
    
    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        print(f"📊 Total signals generated: {len(agent.trades)}")
        print("✅ Agent stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())