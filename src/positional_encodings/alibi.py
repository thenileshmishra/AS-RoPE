import math

import torch
from torch import nn


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def _get_slopes_power_of_two(n_heads: int) -> list[float]:
    start = 2 ** (-(2 ** (-(math.log2(n_heads) - 3))))
    ratio = start
    return [start * (ratio**i) for i in range(n_heads)]


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    if n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {n_heads}")

    if _is_power_of_two(n_heads):
        slopes = _get_slopes_power_of_two(n_heads)
    else:
        closest_power_of_two = 2 ** math.floor(math.log2(n_heads))
        slopes = _get_slopes_power_of_two(closest_power_of_two)
        extra = _get_slopes_power_of_two(2 * closest_power_of_two)
        slopes += extra[0::2][: n_heads - closest_power_of_two]
    return torch.tensor(slopes, dtype=torch.float32)


class ALiBiBias(nn.Module):
    def __init__(self, n_heads: int, max_seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.register_buffer("slopes", get_alibi_slopes(n_heads), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = rel_pos.to(dtype=torch.float32)

        bias = self.slopes.to(device=device).view(1, self.n_heads, 1, 1) * rel_pos.view(
            1, 1, seq_len, seq_len
        )
        return bias.to(dtype=dtype)