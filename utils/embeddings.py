from __future__ import annotations

import math

import torch


def sinusoidal_embedding(t: torch.Tensor, embedding_dim: int = 128) -> torch.Tensor:
	if embedding_dim % 2 != 0:
		raise ValueError("embedding_dim must be even")
	if t.ndim == 0:
		t = t.unsqueeze(0)
	t = t.float().reshape(-1, 1)
	half_dim = embedding_dim // 2
	device = t.device
	if half_dim == 1:
		frequencies = torch.ones(1, device=device)
	else:
		steps = torch.arange(half_dim, device=device, dtype=t.dtype)
		frequencies = torch.exp(-steps * math.log(10000.0) / (half_dim - 1))
	angles = t * frequencies.unsqueeze(0)
	return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
