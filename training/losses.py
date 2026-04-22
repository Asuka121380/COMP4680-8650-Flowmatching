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


def _safe_t(t: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
	return torch.clamp(t, min=eps)


def convert_prediction_space(
	prediction: torch.Tensor,
	*,
	prediction_type: str,
	to_space: str,
	z_t: torch.Tensor,
	t: torch.Tensor,
	t_clip_eps: float = 1e-5,
) -> torch.Tensor:
	if prediction_type not in ("x", "v"):
		raise ValueError("prediction_type must be 'x' or 'v'")
	if to_space not in ("x", "v"):
		raise ValueError("to_space must be 'x' or 'v'")
	if prediction_type == to_space:
		return prediction
	t_safe = _safe_t(t, eps=t_clip_eps)[:, None]
	if prediction_type == "x" and to_space == "v":
		# z_t = x + t * v -> v = (z_t - x) / t
		return (z_t - prediction) / t_safe
	# prediction_type == "v" and to_space == "x"
	# z_t = x + t * v -> x = z_t - t * v
	return z_t - t_safe * prediction


def model_output_to_velocity(
	model_output: torch.Tensor,
	*,
	prediction_type: str,
	z_t: torch.Tensor,
	t: torch.Tensor,
	t_clip_eps: float = 1e-5,
) -> torch.Tensor:
	return convert_prediction_space(
		model_output,
		prediction_type=prediction_type,
		to_space="v",
		z_t=z_t,
		t=t,
		t_clip_eps=t_clip_eps,
	)


def flow_matching_loss(
	model,
	x: torch.Tensor,
	*,
	prediction_type: str,
	loss_type: str,
	t_clip_eps: float = 1e-5,
) -> torch.Tensor:
	if prediction_type not in ("x", "v"):
		raise ValueError("prediction_type must be 'x' or 'v'")
	if loss_type not in ("x", "v"):
		raise ValueError("loss_type must be 'x' or 'v'")
	z_t, t, _, target_v = sample_flow_matching_batch(x)
	target_x = x
	target = target_x if loss_type == "x" else target_v
	model_output = model(z_t, t)
	pred_for_loss = convert_prediction_space(
		model_output,
		prediction_type=prediction_type,
		to_space=loss_type,
		z_t=z_t,
		t=t,
		t_clip_eps=t_clip_eps,
	)
	return torch.mean((pred_for_loss - target) ** 2)


def v_prediction_loss(model, x: torch.Tensor) -> torch.Tensor:
	# Backward-compatible wrapper used by part1.
	return flow_matching_loss(model, x, prediction_type="v", loss_type="v")
