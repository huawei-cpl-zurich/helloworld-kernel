"""A minimal, non-mutating PyTorch reference for manual Cannery registration."""

import torch


def reference(x: torch.Tensor) -> torch.Tensor:
    """Return x + 1, preserving its shape, dtype and device for the float32 cases."""
    return x + 1
