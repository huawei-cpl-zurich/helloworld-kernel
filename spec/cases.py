"""Three deterministic CPU float32 cases, including unusual tensor layouts."""

from collections.abc import Iterator

import torch


def generate_cases() -> Iterator[dict[str, torch.Tensor]]:
    yield {"x": torch.tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.float32)}
    yield {"x": torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)}
    yield {"x": torch.empty((0, 3), dtype=torch.float32)}
