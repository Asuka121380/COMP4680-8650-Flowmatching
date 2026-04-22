from __future__ import annotations

import torch


@torch.no_grad()
def euler_sample(
	model,
	num_samples: int,
	data_dim: int,
	steps: int = 50,
	device: torch.device | str | None = None,
	t_start: float = 1.0,
	t_end: float = 0.0,
	clip_t: float | None = None,
) -> torch.Tensor:
	if device is None:
		device = next(model.parameters()).device
	device = torch.device(device)
	z = torch.randn(num_samples, data_dim, device=device)
	times = torch.linspace(t_start, t_end, steps + 1, device=device)
	for i in range(steps):
		t = times[i]
		next_t = times[i + 1]
		if clip_t is not None:
			t = torch.clamp(t, clip_t, 1.0 - clip_t)
			next_t = torch.clamp(next_t, clip_t, 1.0 - clip_t)
		velocity = model(z, torch.full((num_samples,), t, device=device))
		z = z + (next_t - t) * velocity
	return z
