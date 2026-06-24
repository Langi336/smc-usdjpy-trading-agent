# src/chart_generation/chart_composer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon
import mplfinance as mpf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import io
import base64
from datetime import datetime
import seaborn as sns

class ChartComposer:
    def __init__(self, config: Dict):
        self.config = config
        self.fig_size = config.get('fig_size', (16, 10))
        self.style = config.get('style', 'charles')
        
    def compose_chart(self, price_data: pd.DataFrame, smc_data: Dict, 
                     signal: Optional[Dict] = None) -> io.BytesIO:
        """Compose annotated chart with SMC overlays"""
        fig, axes = self._create_figure()
        ax = axes[0]
        
        # Plot price data
        self._plot_price(ax, price_data)
        
        # Add SMC overlays
        self._plot_order_blocks(ax, smc_data.get('order_blocks', []))
        self._plot_fvgs(ax, smc_data.get('fvgs', []))
        self._plot_liquidity_zones(ax, smc_data.get('liquidity_zones', []))
        self._plot_structure(ax, smc_data.get('structure', {}))
        
        # Add signal if provided
        if signal:
            self._plot_signal(ax, price_data, signal)
        
        # Add legend and annotations
        self._add_legend(ax)
        self._add_annotations(ax, price_data, signal)
        
        # Set title and labels
        ax.set_title(f"USD/JPY - {self.config.get('timeframe', '1H')} Chart", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer
    
    def _create_figure(self):
        """Create figure with subplots"""
        fig = plt.figure(figsize=self.fig_size)
        # Main price chart
        ax = plt.subplot2grid((10, 1), (0, 0), rowspan=8)
        # Volume subplot
        ax_vol = plt.subplot2grid((10, 1), (8, 0), rowspan=2, sharex=ax)
        return fig, (ax, ax_vol)
    
    def _plot_price(self, ax, price_data: pd.DataFrame):
        """Plot price data with custom style"""
        # Use mplfinance for candlestick chart
        mpf.plot(
            price_data,
            type='candle',
            style=self.style,
            ax=ax,
            volume=False,
            show_nontrading=False,
            datetime_format='%H:%M',
            tight_layout=True
        )
        
        # Add moving averages if configured
        if self.config.get('show_moving_averages', True):
            for period, color in [(20, 'blue'), (50, 'orange'), (200, 'red')]:
                ma = price_data['close'].rolling(period).mean()
                ax.plot(price_data.index, ma, color=color, linewidth=1, 
                       alpha=0.7, label=f'MA{period}')
    
    def _plot_order_blocks(self, ax, order_blocks: List):
        """Plot order blocks on chart"""
        for ob in order_blocks:
            if ob.type == 'bullish':
                color = 'green' if not ob.mitigated else 'lightgreen'
                alpha = 0.4 if not ob.mitigated else 0.2
                label = 'Bullish OB' if not ob.mitigated else 'Mitigated OB'
            else:
                color = 'red' if not ob.mitigated else 'lightcoral'
                alpha = 0.4 if not ob.mitigated else 0.2
                label = 'Bearish OB' if not ob.mitigated else 'Mitigated OB'
            
            # Create rectangle for order block
            rect = Rectangle(
                (ob.start_time, ob.start_price),
                ob.end_time - ob.start_time,
                ob.end_price - ob.start_price,
                facecolor=color,
                alpha=alpha,
                linewidth=0
            )
            ax.add_patch(rect)
            
            # Add label
            if not ob.mitigated:
                ax.text(
                    ob.start_time,
                    ob.start_price,
                    'OB',
                    fontsize=8,
                    color='black',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7)
                )
    
    def _plot_fvgs(self, ax, fvgs: List):
        """Plot Fair Value Gaps"""
        for fvg in fvgs:
            if fvg.type == 'bullish':
                color = 'blue'
                label = 'Bullish FVG'
            else:
                color = 'orange'
                label = 'Bearish FVG'
            
            # Create rectangle for FVG
            rect = Rectangle(
                (fvg.start_time, fvg.bottom),
                fvg.end_time - fvg.start_time,
                fvg.top - fvg.bottom,
                facecolor=color,
                alpha=0.2,
                hatch='///',
                linewidth=1,
                edgecolor=color
            )
            ax.add_patch(rect)
            
            # Add fill percentage label
            if fvg.fill_percentage > 0:
                ax.text(
                    fvg.start_time,
                    (fvg.top + fvg.bottom) / 2,
                    f'Fill: {fvg.fill_percentage:.0%}',
                    fontsize=7,
                    color=color,
                    ha='center'
                )
    
    def _plot_liquidity_zones(self, ax, liquidity_zones: List):
        """Plot liquidity zones"""
        for lz in liquidity_zones:
            if lz.type == 'buy_sweep':
                color = 'green'
                label = 'Buy Liquidity'
                marker = '^'
            else:
                color = 'red'
                label = 'Sell Liquidity'
                marker = 'v'
            
            # Plot liquidity level
            ax.axhline(
                y=lz.price_level,
                color=color,
                linestyle='--',
                alpha=0.5,
                linewidth=1
            )
            
            # Add marker
            ax.plot(
                lz.timestamp,
                lz.price_level,
                marker=marker,
                color=color,
                markersize=10,
                alpha=0.7
            )
            
            # Add label
            ax.text(
                lz.timestamp,
                lz.price_level,
                f'Liquidity {lz.type}',
                fontsize=7,
                color=color,
                rotation=90,
                verticalalignment='bottom'
            )
    
    def _plot_structure(self, ax, structure: Dict):
        """Plot market structure"""
        # Plot swing highs
        for sh in structure.get('swing_highs', []):
            ax.plot(
                sh.timestamp,
                sh.price,
                marker='v',
                color='red',
                markersize=8,
                alpha=0.7
            )
        
        # Plot swing lows
        for sl in structure.get('swing_lows', []):
            ax.plot(
                sl.timestamp,
                sl.price,
                marker='^',
                color='green',
                markersize=8,
                alpha=0.7
            )
        
        # Draw trend lines
        if structure.get('trend') == 'bullish':
            # Draw ascending trendline
            if structure.get('swing_lows'):
                lows = structure['swing_lows']
                if len(lows) >= 2:
                    ax.plot(
                        [lows[0].timestamp, lows[-1].timestamp],
                        [lows[0].price, lows[-1].price],
                        color='green',
                        linewidth=2,
                        alpha=0.5,
                        label='Bullish Trendline'
                    )
        elif structure.get('trend') == 'bearish':
            # Draw descending trendline
            if structure.get('swing_highs'):
                highs = structure['swing_highs']
                if len(highs) >= 2:
                    ax.plot(
                        [highs[0].timestamp, highs[-1].timestamp],
                        [highs[0].price, highs[-1].price],
                        color='red',
                        linewidth=2,
                        alpha=0.5,
                        label='Bearish Trendline'
                    )
    
    def _plot_signal(self, ax, price_data: pd.DataFrame, signal: Dict):
        """Plot trade signal with entry, SL, and TP levels"""
        entry_zone = signal['entry_zone']
        stop_loss = signal['stop_loss']
        tp1, tp2, tp3 = signal['take_profit_1'], signal['take_profit_2'], signal['take_profit_3']
        
        # Get current price
        current_price = price_data.iloc[-1]['close']
        current_time = price_data.index[-1]
        
        # Entry zone
        ax.axhspan(
            entry_zone[0], entry_zone[1],
            facecolor='yellow',
            alpha=0.3,
            label='Entry Zone'
        )
        
        # Stop loss
        ax.axhline(
            y=stop_loss,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Stop Loss'
        )
        
        # Take profits
        for tp, label, color in [(tp1, 'TP1', 'green'), 
                                 (tp2, 'TP2', 'blue'), 
                                 (tp3, 'TP3', 'purple')]:
            ax.axhline(
                y=tp,
                color=color,
                linestyle='--',
                linewidth=1.5,
                alpha=0.8
            )
            ax.text(
                current_time,
                tp,
                label,
                fontsize=8,
                color=color,
                fontweight='bold'
            )
        
        # Entry arrow
        arrow_start = current_time
        arrow_end = current_time + pd.Timedelta(minutes=30)
        
        if signal['direction'] == 'long':
            ax.annotate(
                'ENTRY',
                xy=(arrow_end, (entry_zone[0] + entry_zone[1]) / 2),
                xytext=(arrow_start, (entry_zone[0] + entry_zone[1]) / 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10,
                color='green',
                fontweight='bold'
            )
            # Add bullish signal indicator
            ax.scatter(
                current_time,
                current_price,
                marker='^',
                color='green',
                s=100,
                alpha=0.8,
                label='Entry Signal'
            )
        else:
            ax.annotate(
                'ENTRY',
                xy=(arrow_end, (entry_zone[0] + entry_zone[1]) / 2),
                xytext=(arrow_start, (entry_zone[0] + entry_zone[1]) / 2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10,
                color='red',
                fontweight='bold'
            )
            ax.scatter(
                current_time,
                current_price,
                marker='v',
                color='red',
                s=100,
                alpha=0.8,
                label='Entry Signal'
            )
    
    def _add_legend(self, ax):
        """Add legend to chart"""
        # Get legend elements
        handles, labels = ax.get_legend_handles_labels()
        
        # Add custom legend
        if handles:
            ax.legend(
                handles,
                labels,
                loc='upper left',
                bbox_to_anchor=(1.01, 1),
                fontsize=8
            )
    
    def _add_annotations(self, ax, price_data: pd.DataFrame, signal: Optional[Dict] = None):
        """Add explanatory annotations to chart"""
        # Add watermark
        ax.text(
            0.02, 0.98,
            'SMC Analysis | USD/JPY',
            transform=ax.transAxes,
            fontsize=10,
            color='gray',
            alpha=0.6,
            fontweight='bold'
        )
        
        # Add timestamp
        ax.text(
            0.02, 0.02,
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            transform=ax.transAxes,
            fontsize=8,
            color='gray',
            alpha=0.6
        )
        
        # Add confidence if signal provided
        if signal and 'confidence' in signal:
            ax.text(
                0.98, 0.98,
                f'Confidence: {signal["confidence"]:.0%}',
                transform=ax.transAxes,
                fontsize=10,
                color='blue',
                fontweight='bold',
                horizontalalignment='right'
            )
        
        # Add RR ratio if signal provided
        if signal and 'rr_ratio' in signal:
            ax.text(
                0.98, 0.92,
                f'Risk/Reward: {signal["rr_ratio"]:.1f}:1',
                transform=ax.transAxes,
                fontsize=10,
                color='orange',
                fontweight='bold',
                horizontalalignment='right'
            )