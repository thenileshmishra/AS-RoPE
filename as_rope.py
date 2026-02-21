import torch
from torch import nn


class ASRotaryEmbedding(nn.Module):
    """
    Adaptive Spectral RoPE (AS-RoPE).

    Uses a shared learnable gate vector (freq_gates) to scale base rotary
    frequencies before computing phase:
        theta = position * base_freqs * freq_gates
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for AS-RoPE, got {d_model}")
        if d_model != n_heads * head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()

        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)

    def _build_freqs_cis(self, seq_len: int, freq_gates: torch.Tensor, device: torch.device) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        if freq_gates.dim() != 1 or freq_gates.numel() != self.d_model // 2:
            raise ValueError(
                f"freq_gates must have shape ({self.d_model // 2},), got {tuple(freq_gates.shape)}"
            )

        gates = freq_gates.view(self.n_heads, self.head_dim // 2).to(device=device, dtype=torch.float32)
        pos = self.positions[:seq_len].to(device=device, dtype=torch.float32)
        base = self.base_freqs.to(device=device, dtype=torch.float32)

        theta = pos[:, None, None] * base[None, None, :] * gates[None, :, :]
        freqs_cis = torch.polar(torch.ones_like(theta), theta)
        return freqs_cis.permute(1, 0, 2)

    def apply_rotary(self, x: torch.Tensor, seq_len: int, freq_gates: torch.Tensor) -> torch.Tensor:
        freqs_cis = self._build_freqs_cis(seq_len, freq_gates, x.device)

        x_pair = x.float().view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_pair)

        freqs_cis = freqs_cis.unsqueeze(0)
        x_rot = x_complex * freqs_cis

        x_out = torch.view_as_real(x_rot).flatten(start_dim=-2)
        return x_out.type_as(x)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, freq_gates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        return (
            self.apply_rotary(q, seq_len, freq_gates),
            self.apply_rotary(k, seq_len, freq_gates),
        )
