"""
Unit and Integration Tests for Benchmark Models and Pipelines.
"""

import pytest
import torch
import numpy as np
from benchmarks.models.mult_lite import CrossModalAttentionNetwork
from benchmarks.models.early_fusion import MultimodalEarlyFusionNetwork

def test_mult_lite_forward_and_backward():
    model = CrossModalAttentionNetwork(audio_dim=1624, text_dim=768, hidden_dim=256, num_classes=5)
    audio = torch.randn(4, 1624)
    text = torch.randn(4, 768)
    logits = model(audio, text)
    assert logits.shape == (4, 5)
    loss = logits.sum()
    loss.backward()
    assert model.audio_proj.weight.grad is not None
    assert model.text_proj.weight.grad is not None

def test_early_fusion_forward_and_backward():
    model = MultimodalEarlyFusionNetwork(audio_dim=1624, text_dim=768, hidden_dim=256, num_classes=5)
    audio = torch.randn(4, 1624)
    text = torch.randn(4, 768)
    logits = model(audio, text)
    assert logits.shape == (4, 5)
    loss = logits.sum()
    loss.backward()
    assert model.audio_proj.weight.grad is not None
    assert model.text_proj.weight.grad is not None
