from __future__ import annotations

import torch


def sample_flow_matching_batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	batch_size = x.shape[0]
	device = x.device
	t = torch.rand(batch_size, device=device)
	eps = torch.randn_like(x)
	z_t = (1.0 - t[:, None]) * x + t[:, None] * eps
	target_v = eps - x
	return z_t, t, eps, target_v


def v_prediction_loss(model, x: torch.Tensor) -> torch.Tensor:
	z_t, t, _, target_v = sample_flow_matching_batch(x)
	pred_v = model(z_t, t)
	return torch.mean((pred_v - target_v) ** 2)
