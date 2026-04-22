from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import v_prediction_loss


@dataclass
class TrainResult:
	model: nn.Module
	losses: list[float]
	checkpoint_path: Path | None = None


def train_model(
	model: nn.Module,
	dataloader: DataLoader,
	*,
	steps: int = 25_000,
	learning_rate: float = 1e-3,
	device: torch.device | str | None = None,
	log_every: int = 500,
	checkpoint_path: Path | None = None,
	max_grad_norm: float | None = None,
) -> TrainResult:
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device(device)
	model = model.to(device)
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	losses: list[float] = []
	iterator = iter(dataloader)
	progress = tqdm(range(steps), desc="train", leave=False)
	for step in progress:
		try:
			batch = next(iterator)
		except StopIteration:
			iterator = iter(dataloader)
			batch = next(iterator)
		batch = batch.to(device).float()
		optimizer.zero_grad(set_to_none=True)
		loss = v_prediction_loss(model, batch)
		loss.backward()
		if max_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
		optimizer.step()
		loss_value = float(loss.item())
		losses.append(loss_value)
		if log_every > 0 and (step + 1) % log_every == 0:
			progress.set_postfix(loss=f"{loss_value:.4f}")
	if checkpoint_path is not None:
		checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
	return TrainResult(model=model, losses=losses, checkpoint_path=checkpoint_path)
