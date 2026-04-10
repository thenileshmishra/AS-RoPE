"""Smoke tests for sinusoidal positional encoding and model integration."""

import torch

from src.model import GPT
from src.positional_encodings.sinusoidal import SinusoidalPositionalEmbedding


def test_sinusoidal_shape():
    """Output tensor has correct shape [seq_len, d_model]."""
    pe = SinusoidalPositionalEmbedding(d_model=64, max_seq_len=128)
    out = pe(seq_len=32)
    assert out.shape == (32, 64)
    out_full = pe(seq_len=128)
    assert out_full.shape == (128, 64)


def test_sinusoidal_deterministic():
    """Same parameters always produce the same output."""
    pe1 = SinusoidalPositionalEmbedding(d_model=64, max_seq_len=128)
    pe2 = SinusoidalPositionalEmbedding(d_model=64, max_seq_len=128)
    out1 = pe1(32)
    out2 = pe2(32)
    assert torch.equal(out1, out2)


def test_sinusoidal_values_bounded():
    """Sinusoidal values are in [-1, 1]."""
    pe = SinusoidalPositionalEmbedding(d_model=256, max_seq_len=512)
    out = pe(512)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_model_forward_sinusoidal():
    """GPT forward pass works with positional_encoding='sinusoidal'."""
    model = GPT(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=32,
        positional_encoding="sinusoidal",
    )
    model.eval()
    x = torch.randint(0, 100, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, 100)
    assert loss is None


def test_model_forward_sinusoidal_with_targets():
    """Loss is finite when targets are provided."""
    model = GPT(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=32,
        positional_encoding="sinusoidal",
    )
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 16, 100)
    assert loss is not None
    assert torch.isfinite(loss)


def test_model_rope_unchanged():
    """Verify rope still works after sinusoidal addition (no regression)."""
    model = GPT(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=32,
        positional_encoding="rope",
    )
    x = torch.randint(0, 100, (2, 16))
    logits, _ = model(x)
    assert logits.shape == (2, 16, 100)


if __name__ == "__main__":
    test_sinusoidal_shape()
    test_sinusoidal_deterministic()
    test_sinusoidal_values_bounded()
    test_model_forward_sinusoidal()
    test_model_forward_sinusoidal_with_targets()
    test_model_rope_unchanged()
    print("All sinusoidal smoke tests passed.")
