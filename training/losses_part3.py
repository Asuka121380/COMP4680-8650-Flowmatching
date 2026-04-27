from __future__ import annotations

import math

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


def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	return torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


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
		return (z_t - prediction) / t_safe
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


def _apply_target_scaling(
	target: torch.Tensor,
	*,
	mode: str,
	target_space: str,
) -> torch.Tensor:
	if mode == "none":
		return target
	if mode == "sqrt_dim":
		if target_space != "v":
			return target
		return target / math.sqrt(target.shape[-1])
	if mode == "norm":
		return target / _safe_norm(target)
	raise ValueError(f"Unknown target_scaling_mode: {mode}")


def _compute_sample_weights(
	target: torch.Tensor,
	*,
	mode: str,
	target_space: str,
) -> torch.Tensor:
	batch_size = target.shape[0]
	device = target.device
	dtype = target.dtype
	if mode == "none":
		return torch.ones(batch_size, device=device, dtype=dtype)
	if mode == "dim":
		if target_space != "v":
			return torch.ones(batch_size, device=device, dtype=dtype)
		return torch.full((batch_size,), 1.0 / target.shape[-1], device=device, dtype=dtype)
	if mode == "inv_target_norm":
		return (1.0 / _safe_norm(target).squeeze(-1)).to(dtype=dtype)
	if mode == "inv_target_norm_sq":
		norm = _safe_norm(target).squeeze(-1)
		return (1.0 / (norm * norm)).to(dtype=dtype)
	raise ValueError(f"Unknown loss_normalization_mode: {mode}")


def _compute_time_weights(t: torch.Tensor, *, mode: str, eps: float = 1e-5) -> torch.Tensor:
	if mode == "none":
		return torch.ones_like(t)
	t_safe = torch.clamp(t, min=eps, max=1.0)
	if mode == "t":
		return t_safe
	if mode == "1_minus_t":
		return 1.0 - t_safe
	if mode == "inv_t":
		return 1.0 / t_safe
	if mode == "inv_1_minus_t":
		return 1.0 / torch.clamp(1.0 - t_safe, min=eps)
	raise ValueError(f"Unknown time_weighting_mode: {mode}")


def flow_matching_loss(
	model,
	x: torch.Tensor,
	*,
	prediction_type: str,
	loss_type: str,
	t_clip_eps: float = 1e-5,
	target_scaling_mode: str = "none",
	loss_normalization_mode: str = "none",
	time_weighting_mode: str = "none",
) -> torch.Tensor:
	if prediction_type not in ("x", "v"):
		raise ValueError("prediction_type must be 'x' or 'v'")
	if loss_type not in ("x", "v"):
		raise ValueError("loss_type must be 'x' or 'v'")

	z_t, t, _, target_v = sample_flow_matching_batch(x)
	target_x = x
	target = target_x if loss_type == "x" else target_v
	target = _apply_target_scaling(target, mode=target_scaling_mode, target_space=loss_type)

	model_output = model(z_t, t)
	pred_for_loss = convert_prediction_space(
		model_output,
		prediction_type=prediction_type,
		to_space=loss_type,
		z_t=z_t,
		t=t,
		t_clip_eps=t_clip_eps,
	)
	pred_for_loss = _apply_target_scaling(pred_for_loss, mode=target_scaling_mode, target_space=loss_type)

	residual = pred_for_loss - target
	per_sample_loss = torch.mean(residual * residual, dim=-1)

	sample_weights = _compute_sample_weights(
		target,
		mode=loss_normalization_mode,
		target_space=loss_type,
	)
	time_weights = _compute_time_weights(t, mode=time_weighting_mode, eps=t_clip_eps)
	weighted_loss = per_sample_loss * sample_weights * time_weights
	return torch.mean(weighted_loss)


def v_prediction_loss(model, x: torch.Tensor) -> torch.Tensor:
	return flow_matching_loss(model, x, prediction_type="v", loss_type="v")
