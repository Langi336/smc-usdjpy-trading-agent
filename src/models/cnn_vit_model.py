# src/models/cnn_vit_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional
import numpy as np

class CNNViTModel(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 128)
        self.seq_len = config.get('seq_len', 96)
        self.pred_len = config.get('pred_len', 24)
        
        # CNN for pattern extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Vision Transformer for pattern recognition
        self.vit = VisionTransformer(
            image_size=96,
            patch_size=8,
            num_classes=self.pred_len,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=512
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.pred_len)
        )
        
    def forward(self, x):
        # CNN forward
        cnn_input = x.unsqueeze(1)  # Add channel dimension
        cnn_features = self.cnn_layers(cnn_input)
        cnn_features = cnn_features.squeeze(-1)
        
        # ViT forward
        vit_input = x.view(x.shape[0], 1, -1)  # Reshape for ViT
        vit_features = self.vit(vit_input)
        
        # Fusion
        combined = torch.cat([cnn_features, vit_features], dim=-1)
        predictions = self.fusion(combined)
        
        return {
            'predictions': predictions.unsqueeze(-1),
            'cnn_features': cnn_features,
            'vit_features': vit_features
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

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            1, dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, dim)
        )
        
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=depth
        )
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        # Patch embedding
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embedding and class token
        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)
        x = x + self.position_embedding
        
        # Transformer
        x = self.transformer(x)
        
        # MLP head
        features = x[:, 0]  # Class token
        output = self.mlp_head(features)
        
        return output