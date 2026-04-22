from __future__ import annotations

import torch

from training.losses import model_output_to_velocity


@torch.no_grad()
def euler_sample(
	model,
	num_samples: int,
	data_dim: int,
	prediction_type: str = "v",
	steps: int = 50,
	device: torch.device | str | None = None,
	t_start: float = 1.0,
	t_end: float = 0.0,
	clip_t: float | None = None,
	t_clip_eps: float = 1e-5,
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
		t_batch = torch.full((num_samples,), t, device=device)
		model_output = model(z, t_batch)
		velocity = model_output_to_velocity(
			model_output,
			prediction_type=prediction_type,
			z_t=z,
			t=t_batch,
			t_clip_eps=t_clip_eps,
		)
		z = z + (next_t - t) * velocity
	return z
