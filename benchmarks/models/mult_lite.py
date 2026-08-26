"""
MulT-Lite: Multimodal Cross-Modal Attention Network for Emotion Recognition.
Implements 4-head cross-modal multi-head attention (Text Query, Audio Key/Value).
"""

import torch
import torch.nn as nn

class CrossModalAttentionNetwork(nn.Module):
    def __init__(
        self,
        audio_dim: int = 1624,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 5,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Modality Projections to shared latent space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.layer_norm_a = nn.LayerNorm(hidden_dim)
        self.layer_norm_t = nn.LayerNorm(hidden_dim)

        # Cross-Modal Multi-Head Attention (Query: Text, Key: Audio, Value: Audio)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        # audio_feat: (Batch, audio_dim), text_feat: (Batch, text_dim)
        a_proj = self.layer_norm_a(self.audio_proj(audio_feat)).unsqueeze(1)  # (B, 1, H)
        t_proj = self.layer_norm_t(self.text_proj(text_feat)).unsqueeze(1)   # (B, 1, H)

        # Text queries acoustic representation
        attn_out, _ = self.cross_attention(
            query=t_proj,
            key=a_proj,
            value=a_proj
        )

        attn_out = attn_out.squeeze(1)  # (B, H)
        t_proj = t_proj.squeeze(1)      # (B, H)

        # Fuse attended acoustic context with textual representation
        fused = torch.cat([attn_out, t_proj], dim=1)  # (B, 2*H)
        logits = self.classifier(fused)
        return logits
