"""Toy split LoRA module and training demo.

This script defines a ``SplitLoRALinear`` layer that splits its input features
into several chunks, attaching a low‑rank adapter (LoRA) to each chunk. The base
linear weight is frozen while the adapter weights are trainable. A small demo
at the bottom shows the layer learning to sum its inputs.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class SplitLoRALinear(nn.Module):
    """Linear layer augmented with split LoRA adapters.

    Parameters
    ----------
    in_features: int
        Number of input features.
    out_features: int
        Number of output features.
    r: int
        Rank of the LoRA adapters.
    n_splits: int
        Number of equal-sized input chunks, each with its own adapter.
    alpha: float
        Scaling factor applied to the adapter output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        n_splits: int = 2,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features % n_splits != 0:
            raise ValueError("in_features must be divisible by n_splits")
        self.in_features = in_features
        self.out_features = out_features
        self.n_splits = n_splits
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Base weight is frozen like in standard LoRA setups.
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # One low-rank adapter (A and B) per split.
        split_size = in_features // n_splits
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        for _ in range(n_splits):
            a = nn.Parameter(torch.empty(r, split_size))
            b = nn.Parameter(torch.empty(out_features, r))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            nn.init.zeros_(b)
            self.A.append(a)
            self.B.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.t()
        splits = x.split(x.size(-1) // self.n_splits, dim=-1)
        for split, A, B in zip(splits, self.A, self.B):
            out = out + (split @ A.t() @ B.t()) * self.scaling
        return out


def demo() -> None:
    """Train a SplitLoRALinear layer to learn the sum of inputs."""
    torch.manual_seed(0)
    layer = SplitLoRALinear(8, 1, r=2, n_splits=4, alpha=2.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)

    for step in range(200):
        x = torch.randn(32, 8)
        y = x.sum(dim=1, keepdim=True)
        pred = layer(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Final training loss: {loss.item():.4f}")


if __name__ == "__main__":
    demo()
