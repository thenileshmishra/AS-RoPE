import torch
from torch import nn


class ScaledRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()

        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)

    def _build_freqs_cis(self, seq_len: int, gamma: torch.Tensor, device: torch.device) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        if gamma.dim() != 0:
            raise ValueError(f"gamma must be a scalar tensor, got shape {tuple(gamma.shape)}")

        pos = self.positions[:seq_len].to(device=device, dtype=torch.float32)
        base = self.base_freqs.to(device=device, dtype=torch.float32)
        theta = pos[:, None] * base[None, :] * gamma.to(device=device, dtype=torch.float32)
        return torch.polar(torch.ones_like(theta), theta)

    def apply_rotary(self, x: torch.Tensor, seq_len: int, gamma: torch.Tensor) -> torch.Tensor:
        freqs_cis = self._build_freqs_cis(seq_len, gamma, x.device)

        x_pair = x.float().view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_pair)

        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
        x_rot = x_complex * freqs_cis

        x_out = torch.view_as_real(x_rot).flatten(start_dim=-2)
        return x_out.type_as(x)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        return self.apply_rotary(q, seq_len, gamma), self.apply_rotary(k, seq_len, gamma)
