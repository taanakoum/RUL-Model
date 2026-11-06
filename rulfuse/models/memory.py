"""A causal recurrent memory over a unit's whole history (PROJECT_PLAN.md D21).

A model that reads a fixed window of recent steps (B6's 96 OmniFuser steps) forgets
everything older. ``CausalMemory`` carries it: a GRU (or LSTM) runs over the unit's step
inputs from its first step, and the state it has at step t summarizes every step up to t.

To keep a long history cheap (B6: up to about 5,000 one-second steps per tool), the steps
are averaged in blocks of ``block`` steps and the recurrent cell runs over the blocks. Step
t reads the state after the last block that is **complete at t**, block j = (t + 1) // block
− 1; the steps of the unfinished block are the most recent ones, which the window model
sees. Before the first complete block the state is zero. So the state used at step t never
depends on a step after t (tested).

No normalization is applied to the block means: the step inputs are already standardized
(on the training units), and a LayerNorm across features would cancel a rise shared by all
of them, such as the common rise of the force descriptors with wear.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

CELLS = ("gru", "lstm")


class CausalMemory(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, block: int = 1, cell: str = "gru"):
        super().__init__()
        if block < 1:
            raise ValueError("block must be >= 1")
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}")
        self.block, self.hidden_dim = block, hidden_dim
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(input_dim, hidden_dim, batch_first=True)

    def blocks(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """(U, T, D) step inputs -> (U, ceil(T / block), D) means over each block's valid
        steps (steps at or beyond a unit's length are padding and never averaged in)."""
        units, steps, dim = x.shape
        b = self.block
        n = -(-steps // b)
        valid = torch.ones(units, steps, device=x.device, dtype=x.dtype)
        if lengths is not None:
            valid = (torch.arange(steps, device=x.device)[None] < lengths[:, None]).to(x.dtype)
        pad = n * b - steps
        if pad:
            x = torch.cat([x, x.new_zeros(units, pad, dim)], dim=1)
            valid = torch.cat([valid, valid.new_zeros(units, pad)], dim=1)
        sums = (x * valid[..., None]).reshape(units, n, b, dim).sum(dim=2)
        counts = valid.reshape(units, n, b).sum(dim=2).clamp_min(1.0)
        return sums / counts[..., None]

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """(U, T, D) step inputs (in time order, padded at the end) -> (U, T, hidden): the
        memory each step may read."""
        units, steps, _ = x.shape
        out, _ = self.rnn(self.blocks(x, lengths))  # (U, n_blocks, hidden)
        index = (torch.arange(steps, device=x.device) + 1) // self.block  # 0: no block yet
        padded = torch.cat([out.new_zeros(units, 1, out.shape[-1]), out], dim=1)
        return padded[:, index]
