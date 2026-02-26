import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [..., d] into [..., d/2, 2] pairs and rotate each pair:
    (x_even, x_odd) -> (-x_odd, x_even)
    """
    x_ = x.view(*x.shape[:-1], -1, 2)
    x_even = x_[..., 0]
    x_odd = x_[..., 1]
    out = torch.stack((-x_odd, x_even), dim=-1)
    return out.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as in RoFormer.

    Uses complex rotation on even/odd dimension pairs:
    (x_even + i*x_odd) * exp(i * theta_pos)
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # RoFormer frequency schedule for pair-wise dimensions.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()

        # [max_seq_len, head_dim/2]
        freqs = torch.outer(positions, inv_freq)

        # Complex phase: exp(i * theta) = cos(theta) + i sin(theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _get_freqs_cis(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        return self.freqs_cis[:seq_len].to(device)

    def apply_rotary(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        x: [B, H, T, D]
        Returns: [B, H, T, D] with RoPE applied over even/odd pairs.
        """
        freqs_cis = self._get_freqs_cis(seq_len, x.device)  # [T, D/2]

        # Build complex numbers from even/odd pairs.
        # [B, H, T, D] -> [B, H, T, D/2, 2] -> complex [B, H, T, D/2]
        x_pair = x.float().view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_pair)

        # Broadcast freqs to [1, 1, T, D/2]
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

        # Complex rotation.
        x_rot = x_complex * freqs_cis

        # Back to real tensor with original shape.
        x_out = torch.view_as_real(x_rot).flatten(start_dim=-2)
        return x_out.type_as(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: [B, H, T, D]
        """
        seq_len = q.size(-2)
        return self.apply_rotary(q, seq_len), self.apply_rotary(k, seq_len)
