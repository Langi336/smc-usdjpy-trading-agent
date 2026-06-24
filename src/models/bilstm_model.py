# src/models/bilstm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional

class BiLSTMModel(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.seq_len = config.get('seq_len', 96)
        self.pred_len = config.get('pred_len', 24)
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU()
        )
        
        # Output layers
        self.decoder = nn.Linear(self.hidden_dim // 2, self.pred_len)
        
        # Confidence prediction
        self.confidence_head = nn.Linear(self.hidden_dim // 2, 1)
        
    def forward(self, x):
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention over sequence
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Feature extraction
        features = self.feature_extractor(attended[:, -1, :])
        
        # Predictions
        predictions = self.decoder(features)
        
        # Confidence
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return {
            'predictions': predictions.unsqueeze(-1),
            'confidence': confidence,
            'features': features
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = F.mse_loss(outputs['predictions'], y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-4)
        )
        return optimizer