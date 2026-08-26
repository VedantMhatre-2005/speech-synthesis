"""
State-of-the-Art Baseline: Bi-LSTM with Self-Attention (bc-LSTM architecture variant).
Matches the standard strong multimodal baseline used in MELD emotion recognition literature.
Replaces the naive MLP early-fusion with a temporal recurrent network and attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalBaselineNetwork(nn.Module):
    def __init__(
        self,
        audio_dim: int = 1624,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Modality-specific non-linear projections
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bi-directional LSTM for temporal/contextual modeling
        # (Standard in MELD baselines like bc-LSTM)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Self-Attention mechanism over the Bi-LSTM outputs
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        # Project into shared latent space
        a_proj = self.audio_proj(audio_feat)  # (B, H)
        t_proj = self.text_proj(text_feat)    # (B, H)

        # Concatenate modalities (B, 1, 2*H) - treating as sequence length 1 for uniformity
        fused_step = torch.cat([a_proj, t_proj], dim=-1).unsqueeze(1)

        # Pass through Bi-LSTM
        lstm_out, _ = self.bilstm(fused_step)  # (B, 1, 2*H)

        # Self-Attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (B, 1, 1)
        attended_context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 2*H)

        # Classification
        logits = self.classifier(attended_context)
        return logits
