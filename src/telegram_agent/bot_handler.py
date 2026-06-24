# src/telegram_agent/bot_handler.py
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
import asyncio
from typing import Dict, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramBotHandler:
    def __init__(self, config: Dict, agent_brain, signal_generator, chart_composer):
        self.config = config
        self.agent_brain = agent_brain
        self.signal_generator = signal_generator
        self.chart_composer = chart_composer
        self.token = config['telegram_bot_token']
        self.application = None
        self.pending_signals = {}
        self.user_preferences = {}
        
    async def start(self):
        """Start the Telegram bot"""
        self.application = Application.builder().token(self.token).build()
        
        # Register handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("bias", self.bias_command))
        self.application.add_handler(CommandHandler("levels", self.levels_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("journal", self.journal_command))
        
        # Callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.callback_handler))
        
        # Message handler for Q&A
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            self.message_handler
        ))
        
        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        logger.info("Telegram bot started successfully")
        
    async def stop(self):
        """Stop the Telegram bot"""
        if self.application:
            await self.application.stop()
            await self.application.shutdown()
            
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        welcome_message = """
        🚀 Welcome to the USD/JPY SMC AI Trading Agent!
        
        I analyze the USD/JPY pair using Smart Money Concepts and Deep Learning to provide high-probability trading signals.
        
        Available commands:
        /bias - Get current market bias
        /levels - Get key SMC levels
        /stats - View performance statistics
        /journal - View recent trades
        /ask <question> - Ask about SMC concepts or current analysis
        
        You can also ask questions in natural language!
        
        ⚠️ Disclaimer: This is for educational purposes only. Always do your own research.
        """
        await update.message.reply_text(welcome_message)
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
        📖 Help & Commands
        
        Trading Commands:
        /bias - Current market bias and structure
        /levels - Key support/resistance levels
        /signal - Generate a new trading signal (admin only)
        
        Performance Commands:
        /stats - Trading statistics
        /journal - Recent trade journal
        
        Q&A:
        /ask <question> - Ask about SMC concepts or analysis
        Example: /ask what is an order block?
        
        Support:
        /help - Show this message
        """
        await update.message.reply_text(help_message)
        
    async def bias_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /bias command"""
        # Get current market bias
        bias_analysis = await self._get_market_bias()
        await update.message.reply_text(bias_analysis)
        
    async def levels_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /levels command"""
        # Get key levels
        levels = await self._get_key_levels()
        await update.message.reply_text(levels)
        
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        # Get performance statistics
        stats = await self._get_performance_stats()
        report = self.agent_brain.generate_performance_report(stats)
        await update.message.reply_text(report)
        
    async def journal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /journal command"""
        # Get recent trades
        trades = await self._get_recent_trades()
        journal = self._format_journal(trades)
        await update.message.reply_text(journal)
        
    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user messages (Q&A)"""
        user_message = update.message.text
        user_id = update.effective_user.id
        
        # Check if it's a question
        if user_message.startswith('/ask'):
            question = user_message[4:].strip()
            if question:
                answer = self.agent_brain.answer_question(question, await self._get_context())
                await update.message.reply_text(answer)
            else:
                await update.message.reply_text("Please provide a question. Example: /ask what is FVG?")
        else:
            # Treat as natural language question
            answer = self.agent_brain.answer_question(user_message, await self._get_context())
            await update.message.reply_text(answer)
            
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = update.effective_user.id
        
        if data.startswith('confirm_'):
            signal_id = data.replace('confirm_', '')
            await self._confirm_signal(query, signal_id)
            
        elif data.startswith('reject_'):
            signal_id = data.replace('reject_', '')
            await self._reject_signal(query, signal_id)
            
        elif data.startswith('modify_'):
            # Handle signal modification
            await query.edit_message_text("Signal modification feature coming soon!")
            
    async def send_signal_alert(self, signal: Dict, chart_buffer: bytes):
        """Send signal alert to subscribed users"""
        # Create inline keyboard
        keyboard = [
            [
                InlineKeyboardButton("✅ Confirm", callback_data=f"confirm_{signal.get('id')}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject_{signal.get('id')}")
            ],
            [InlineKeyboardButton("✏️ Modify", callback_data=f"modify_{signal.get('id')}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Generate narrative
        narrative = self.agent_brain.generate_signal_narrative(
            signal, 
            await self._get_smc_data()
        )
        
        # Prepare caption
        caption = self._format_signal_caption(signal, narrative)
        
        # Send to all subscribers
        for user_id in self.config['subscribers']:
            try:
                await self.application.bot.send_photo(
                    chat_id=user_id,
                    photo=chart_buffer,
                    caption=caption,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Failed to send signal to {user_id}: {e}")
                
    def _format_signal_caption(self, signal: Dict, narrative: str) -> str:
        """Format signal alert caption"""
        direction = signal.get('direction', 'UNKNOWN').upper()
        entry = signal.get('entry_zone', (0, 0))
        sl = signal.get('stop_loss', 0)
        tp1, tp2, tp3 = signal.get('take_profit_1', 0), signal.get('take_profit_2', 0), signal.get('take_profit_3', 0)
        rr = signal.get('rr_ratio', 0)
        confidence = signal.get('confidence', 0)
        
        direction_emoji = '🟢' if direction == 'LONG' else '🔴' if direction == 'SHORT' else '⚪'
        
        return f"""
{direction_emoji} *{direction} SIGNAL*

*Entry Zone:* {entry[0]:.4f} - {entry[1]:.4f}
*Stop Loss:* {sl:.4f}
*Take Profit 1:* {tp1:.4f}
*Take Profit 2:* {tp2:.4f}
*Take Profit 3:* {tp3:.4f}
*Risk/Reward:* {rr:.1f}:1
*Confidence:* {confidence:.0%}

{narrative}

---
*⚡ Action Required:*
Tap 'Confirm' to execute, 'Reject' to dismiss, or 'Modify' to adjust.
"""
    
    async def _get_market_bias(self) -> str:
        """Get current market bias"""
        # Get latest SMC analysis
        smc_data = await self._get_smc_data()
        structure = smc_data.get('structure', {})
        trend = structure.get('trend', 'ranging')
        
        # Generate bias analysis
        analysis = f"""
📊 *Current Market Bias*

*Structure:* {trend.upper()}

*Key Levels:*
- Recent High: {structure.get('recent_high', 0):.4f}
- Recent Low: {structure.get('recent_low', 0):.4f}
- Highest High: {structure.get('highest_high', 0):.4f}
- Lowest Low: {structure.get('lowest_low', 0):.4f}

*Order Blocks:*
- Bullish: {len([ob for ob in smc_data.get('order_blocks', []) if ob.type == 'bullish'])}
- Bearish: {len([ob for ob in smc_data.get('order_blocks', []) if ob.type == 'bearish'])}

*FVGs:*
- Bullish: {len([fvg for fvg in smc_data.get('fvgs', []) if fvg.type == 'bullish'])}
- Bearish: {len([fvg for fvg in smc_data.get('fvgs', []) if fvg.type == 'bearish'])}

*Analysis:*
{self.agent_brain._generate_bias_insights(smc_data)}
"""
        return analysis
    
    async def _get_key_levels(self) -> str:
        """Get key support and resistance levels"""
        smc_data = await self._get_smc_data()
        
        # Get all levels
        levels = []
        
        # Structure levels
        structure = smc_data.get('structure', {})
        if structure.get('recent_high'):
            levels.append(('Resistance', structure['recent_high']))
        if structure.get('recent_low'):
            levels.append(('Support', structure['recent_low']))
            
        # Order blocks
        for ob in smc_data.get('order_blocks', []):
            if ob.type == 'bullish':
                levels.append(('Bullish OB', ob.end_price))
            else:
                levels.append(('Bearish OB', ob.start_price))
                
        # FVGs
        for fvg in smc_data.get('fvgs', []):
            if fvg.type == 'bullish':
                levels.append(('Bullish FVG', fvg.bottom))
            else:
                levels.append(('Bearish FVG', fvg.top))
                
        # Sort levels
        levels.sort(key=lambda x: x[1])
        
        # Format levels
        level_text = "📊 *Key SMC Levels*\n\n"
        for level_type, price in levels:
            level_text += f"{level_type}: {price:.4f}\n"
            
        level_text += f"\n*Current Price:* {await self._get_current_price():.4f}"
        level_text += f"\n*Spread:* {await self._get_current_spread():.2f} pips"
        
        return level_text
    
    async def _confirm_signal(self, query, signal_id: str):
        """Handle signal confirmation"""
        signal = self.pending_signals.get(signal_id)
        if signal:
            # Execute trade
            await self._execute_trade(signal)
            await query.edit_message_text(
                f"✅ Signal confirmed and executed!\n\n"
                f"Trade Details:\n"
                f"Direction: {signal['direction'].upper()}\n"
                f"Entry: {signal['entry_zone']}\n"
                f"Stop Loss: {signal['stop_loss']}\n"
                f"Take Profits: {signal['take_profit_1']}, {signal['take_profit_2']}, {signal['take_profit_3']}\n"
                f"Risk/Reward: {signal['rr_ratio']:.1f}:1"
            )
        else:
            await query.edit_message_text("Signal expired or already processed.")
            
    async def _reject_signal(self, query, signal_id: str):
        """Handle signal rejection"""
        if signal_id in self.pending_signals:
            del self.pending_signals[signal_id]
        await query.edit_message_text("❌ Signal rejected.")
        
    def _format_journal(self, trades: List) -> str:
        """Format trade journal"""
        journal = "📖 *Trade Journal*\n\n"
        
        if not trades:
            journal += "No trades yet."
            return journal
            
        for i, trade in enumerate(trades[:10], 1):
            direction = '🟢' if trade['direction'] == 'long' else '🔴' if trade['direction'] == 'short' else '⚪'
            pnl = trade.get('pnl', 0)
            pnl_emoji = '✅' if pnl > 0 else '❌' if pnl < 0 else '➖'
            
            journal += f"{i}. {direction} {trade['direction'].upper()}"
            journal += f" | Entry: {trade.get('entry', 0):.4f}"
            journal += f" | Exit: {trade.get('exit', 0):.4f}"
            journal += f" | {pnl_emoji} P&L: ${pnl:.2f}\n"
            
        return journal
    
    async def _get_smc_data(self) -> Dict:
        """Get current SMC data"""
        # Implement data fetching
        return {}
    
    async def _get_context(self) -> Dict:
        """Get current market context"""
        return {
            'current_price': await self._get_current_price(),
            'smc_data': await self._get_smc_data()
        }
    
    async def _get_current_price(self) -> float:
        """Get current price"""
        # Implement price fetching
        return 150.00
    
    async def _get_current_spread(self) -> float:
        """Get current spread in pips"""
        # Implement spread calculation
        return 0.8
    
    async def _get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        # Implement stats fetching
        return {
            'win_rate': 0.58,
            'avg_rr': 1.8,
            'total_trades': 150,
            'profit': 12500,
            'sharpe': 1.6,
            'max_drawdown': 0.08
        }
    
    async def _get_recent_trades(self) -> List:
        """Get recent trades"""
        # Implement trade fetching
        return []
    
    async def _execute_trade(self, signal: Dict):
        """Execute trade"""
        # Implement trade execution
        pass