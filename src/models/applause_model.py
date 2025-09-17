"""
Neural Network Model for Applause Detection in Political Speeches
Implements LSTM-based architecture inspired by the "Please Clap" paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class ApplauseLSTM(nn.Module):
    """
    LSTM-based neural network for applause detection
    Architecture inspired by the "Please Clap" paper with improvements
    """
    
    def __init__(self, 
                 mfcc_dim: int = 13,
                 other_features_dim: int = 12,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_classes: int = 2):
        super(ApplauseLSTM, self).__init__()
        
        self.mfcc_dim = mfcc_dim
        self.other_features_dim = other_features_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # MFCC processing branch
        self.mfcc_lstm = nn.LSTM(
            input_size=mfcc_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Other features processing
        self.other_features_fc = nn.Sequential(
            nn.Linear(other_features_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for MFCC features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, mfcc: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            mfcc: MFCC features [batch_size, seq_len, mfcc_dim]
            other_features: Other audio features [batch_size, other_features_dim]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size = mfcc.size(0)
        
        # Process MFCC features through LSTM
        mfcc_output, (hidden, cell) = self.mfcc_lstm(mfcc)
        
        # Apply attention mechanism
        attended_output, _ = self.attention(mfcc_output, mfcc_output, mfcc_output)
        
        # Global average pooling over sequence dimension
        mfcc_features = torch.mean(attended_output, dim=1)  # [batch_size, hidden_dim * 2]
        
        # Process other features
        other_features_processed = self.other_features_fc(other_features)
        
        # Concatenate features
        combined_features = torch.cat([mfcc_features, other_features_processed], dim=1)
        
        # Final classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def get_attention_weights(self, mfcc: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization"""
        with torch.no_grad():
            mfcc_output, _ = self.mfcc_lstm(mfcc)
            _, attention_weights = self.attention(mfcc_output, mfcc_output, mfcc_output)
            return attention_weights

class ApplauseTransformer(nn.Module):
    """
    Transformer-based model as an alternative to LSTM
    """
    
    def __init__(self,
                 mfcc_dim: int = 13,
                 other_features_dim: int = 12,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super(ApplauseTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input projection for MFCC
        self.mfcc_projection = nn.Linear(mfcc_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Other features processing
        self.other_features_fc = nn.Sequential(
            nn.Linear(other_features_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, mfcc: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        # Project MFCC to model dimension
        mfcc_projected = self.mfcc_projection(mfcc)
        
        # Add positional encoding
        mfcc_encoded = self.pos_encoding(mfcc_projected)
        
        # Pass through transformer
        transformer_output = self.transformer(mfcc_encoded)
        
        # Global average pooling
        mfcc_features = torch.mean(transformer_output, dim=1)
        
        # Process other features
        other_features_processed = self.other_features_fc(other_features)
        
        # Combine and classify
        combined_features = torch.cat([mfcc_features, other_features_processed], dim=1)
        logits = self.classifier(combined_features)
        
        return logits

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def create_model(model_type: str = 'lstm', **kwargs) -> nn.Module:
    """Factory function to create models"""
    if model_type.lower() == 'lstm':
        return ApplauseLSTM(**kwargs)
    elif model_type.lower() == 'transformer':
        return ApplauseTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the models
    print("Testing ApplauseLSTM model...")
    lstm_model = ApplauseLSTM()
    print(f"LSTM model parameters: {count_parameters(lstm_model):,}")
    
    # Test forward pass
    batch_size, seq_len = 4, 100
    mfcc_input = torch.randn(batch_size, seq_len, 13)
    other_input = torch.randn(batch_size, 12)
    
    lstm_output = lstm_model(mfcc_input, other_input)
    print(f"LSTM output shape: {lstm_output.shape}")
    
    print("\nTesting ApplauseTransformer model...")
    transformer_model = ApplauseTransformer()
    print(f"Transformer model parameters: {count_parameters(transformer_model):,}")
    
    transformer_output = transformer_model(mfcc_input, other_input)
    print(f"Transformer output shape: {transformer_output.shape}")
    
    print("\nModels created successfully! 🎉")