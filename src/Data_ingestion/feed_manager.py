# src/data_ingestion/feed_manager.py
import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class DataIngestionManager:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', 'USD/JPY')
        logger.info(f"DataIngestionManager initialized for {self.symbol}")
        
    def get_latest_price(self):
        """Get simulated price data"""
        return {
            'symbol': self.symbol,
            'price': 150 + random.uniform(-2, 2),
            'timestamp': datetime.now().isoformat(),
            'bid': 149.8 + random.uniform(-0.5, 0.5),
            'ask': 150.2 + random.uniform(-0.5, 0.5)
        }
    
    def get_historical_data(self, timeframe='1H', days=30):
        """Get simulated historical data"""
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        data = pd.DataFrame({
            'open': np.random.normal(150, 2, len(dates)),
            'high': np.random.normal(151, 2, len(dates)),
            'low': np.random.normal(149, 2, len(dates)),
            'close': np.random.normal(150, 2, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates))
        }, index=dates)
        
        return data