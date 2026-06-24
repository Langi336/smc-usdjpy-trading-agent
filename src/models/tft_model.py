# src/models/tft_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoModel, AutoConfig

class TemporalFusionTransformer(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.output_dim = config.get('output_dim', 1)
        self.seq_len = config.get('seq_len', 96)
        self.pred_len = config.get('pred_len', 24)
        
        # LSTM for sequence processing
        self.lstm_encoder = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim * 2,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=self.num_layers
        )
        
        # Attention for variable selection
        self.variable_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Static covariates processing
        self.static_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.hidden_dim * 2)
        )
        
        # Output layers
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim * self.pred_len)
        )
        
        # Quantile outputs for probabilistic forecasting
        self.quantiles = nn.Linear(self.hidden_dim // 2, self.pred_len * 3)  # 0.1, 0.5, 0.9
        
    def forward(self, x, static_covariates=None):
        batch_size, seq_len, features = x.shape
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm_encoder(x)
        
        # Static covariates attention
        if static_covariates is not None:
            static_embed = self.static_encoder(static_covariates)
            static_embed = static_embed.unsqueeze(1).repeat(1, seq_len, 1)
            lstm_out = lstm_out + static_embed
        
        # Variable selection attention
        attended_vars, _ = self.variable_attention(lstm_out, lstm_out, lstm_out)
        
        # Temporal attention
        temporal_out, _ = self.temporal_attention(attended_vars, attended_vars, attended_vars)
        
        # Transformer
        transformed = self.transformer(temporal_out)
        
        # Decode
        decoded = self.decoder(transformed[:, -1, :])
        predictions = decoded.view(batch_size, self.pred_len, self.output_dim)
        
        # Quantile predictions
        quantile_out = self.quantiles(transformed[:, -1, :])
        quantiles = quantile_out.view(batch_size, self.pred_len, 3)
        
        return {
            'predictions': predictions,
            'quantiles': quantiles
        }
    
    def training_step(self, batch, batch_idx):
        x, static, y = batch
        outputs = self.forward(x, static)
        loss = self._calculate_loss(outputs, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def _calculate_loss(self, outputs, targets):
        # Combined loss: MAE + quantile loss
        pred_loss = F.l1_loss(outputs['predictions'], targets)
        
        # Quantile loss
        q_loss = 0
        quantiles = outputs['quantiles']
        for q_idx, q in enumerate([0.1, 0.5, 0.9]):
            q_pred = quantiles[:, :, q_idx:q_idx+1]
            error = targets - q_pred
            q_loss += torch.max(q * error, (q - 1) * error).mean()
        
        return pred_loss + 0.1 * q_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('epochs', 100),
            eta_min=1e-6
        )
        return [optimizer], [scheduler]