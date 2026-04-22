from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from models.model import FlowMatchingMLP
from sampling.euler import euler_sample
from src.dataloader import AVAILABLE_DATASETS, ToyDiffusionDataset
from training.train import train_model


@dataclass
class Part1Config:
	output_dir: Path = Path("outputs") / "part1_v_pred"
	run_name: str | None = None
	data_dir: Path | None = Path("data")
	steps: int = 25_000
	batch_size: int = 1024
	learning_rate: float = 1e-3
	sampling_steps: int = 50
	seed: int = 42
	max_points: int | None = 5_000
	dpi: int = 200
	hidden_dim: int = 256
	time_embedding_dim: int = 128
	num_hidden_layers: int = 5


def _optional_path(value: str | None) -> Path | None:
	if value is None or value == "":
		return None
	return Path(value)


def _resolve_run_dir(base_output_dir: Path, run_name: str | None) -> Path:
	base_output_dir.mkdir(parents=True, exist_ok=True)
	if run_name is None or run_name.strip() == "":
		run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	candidate = base_output_dir / run_name
	if not candidate.exists():
		return candidate
	suffix = 1
	while True:
		next_candidate = base_output_dir / f"{run_name}_{suffix}"
		if not next_candidate.exists():
			return next_candidate
		suffix += 1


def load_config(config_path: Path) -> Part1Config:
	with config_path.open("rb") as f:
		data = tomllib.load(f)
	train_cfg = data.get("train", {})
	data_cfg = data.get("data", {})
	model_cfg = data.get("model", {})
	paths_cfg = data.get("paths", {})
	return Part1Config(
		output_dir=Path(paths_cfg.get("output_dir", "outputs/part1_v_pred")),
		run_name=paths_cfg.get("run_name", None),
		data_dir=_optional_path(paths_cfg.get("data_dir", "data")),
		steps=int(train_cfg.get("steps", 25_000)),
		batch_size=int(train_cfg.get("batch_size", 1024)),
		learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
		sampling_steps=int(train_cfg.get("sampling_steps", 50)),
		seed=int(train_cfg.get("seed", 42)),
		max_points=train_cfg.get("max_points", 5_000),
		dpi=int(train_cfg.get("dpi", 200)),
		hidden_dim=int(model_cfg.get("hidden_dim", 256)),
		time_embedding_dim=int(model_cfg.get("time_embedding_dim", 128)),
		num_hidden_layers=int(model_cfg.get("num_hidden_layers", 5)),
	)


def merge_cli_overrides(config: Part1Config, args: argparse.Namespace) -> Part1Config:
	if args.output_dir is not None:
		config.output_dir = args.output_dir
	if args.data_dir is not None:
		config.data_dir = args.data_dir
	if args.run_name is not None:
		config.run_name = args.run_name
	if args.steps is not None:
		config.steps = args.steps
	if args.batch_size is not None:
		config.batch_size = args.batch_size
	if args.learning_rate is not None:
		config.learning_rate = args.learning_rate
	if args.sampling_steps is not None:
		config.sampling_steps = args.sampling_steps
	if args.seed is not None:
		config.seed = args.seed
	if args.max_points is not None:
		config.max_points = args.max_points
	if args.dpi is not None:
		config.dpi = args.dpi
	if args.hidden_dim is not None:
		config.hidden_dim = args.hidden_dim
	if args.time_embedding_dim is not None:
		config.time_embedding_dim = args.time_embedding_dim
	if args.num_hidden_layers is not None:
		config.num_hidden_layers = args.num_hidden_layers
	return config


def _sample_points(points: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
	if max_points is None or len(points) <= max_points:
		return points
	rng = np.random.default_rng(seed)
	idx = rng.choice(len(points), size=max_points, replace=False)
	return points[idx]


def _plot_comparison(real: np.ndarray, generated: np.ndarray, title: str, save_path: Path, dpi: int) -> None:
	fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
	axes[0].scatter(real[:, 0], real[:, 1], s=5, alpha=0.7, linewidths=0)
	axes[0].set_title("ground truth")
	axes[1].scatter(generated[:, 0], generated[:, 1], s=5, alpha=0.7, linewidths=0)
	axes[1].set_title("generated")
	for ax in axes:
		ax.set_aspect("equal", adjustable="box")
		ax.grid(alpha=0.25)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(save_path, dpi=dpi)
	plt.close(fig)


def run_experiment(config: Part1Config) -> None:
	torch.manual_seed(config.seed)
	np.random.seed(config.seed)
	run_dir = _resolve_run_dir(config.output_dir, config.run_name)
	run_dir.mkdir(parents=True, exist_ok=True)
	for name in AVAILABLE_DATASETS:
		dataset = ToyDiffusionDataset(name=name, dim=2, data_dir=config.data_dir)
		loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
		model = FlowMatchingMLP(
			data_dim=2,
			hidden_dim=config.hidden_dim,
			time_embedding_dim=config.time_embedding_dim,
			num_hidden_layers=config.num_hidden_layers,
		)
		ckpt_path = run_dir / "checkpoints" / f"{name}_vpred.pt"
		result = train_model(
			model,
			loader,
			steps=config.steps,
			learning_rate=config.learning_rate,
			checkpoint_path=ckpt_path,
		)
		model = result.model.eval()
		real = dataset.data.detach().cpu().numpy()
		generated = euler_sample(model, num_samples=len(dataset), data_dim=2, steps=config.sampling_steps)
		real = _sample_points(real, max_points=config.max_points, seed=config.seed)
		generated = _sample_points(generated.detach().cpu().numpy(), max_points=config.max_points, seed=config.seed)
		_plot_comparison(
			real=real,
			generated=generated,
			title=f"{name} | v-prediction D=2",
			save_path=run_dir / f"{name}_comparison.png",
			dpi=config.dpi,
		)
	return run_dir


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Part 3.2 v-prediction flow matching at D=2")
	parser.add_argument("--config", type=Path, default=Path("configs") / "part1_v_pred.toml")
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--data-dir", type=Path, default=None)
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--steps", type=int, default=None)
	parser.add_argument("--batch-size", type=int, default=None)
	parser.add_argument("--learning-rate", type=float, default=None)
	parser.add_argument("--sampling-steps", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--max-points", type=int, default=None)
	parser.add_argument("--dpi", type=int, default=None)
	parser.add_argument("--hidden-dim", type=int, default=None)
	parser.add_argument("--time-embedding-dim", type=int, default=None)
	parser.add_argument("--num-hidden-layers", type=int, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = load_config(args.config)
	config = merge_cli_overrides(config, args)
	run_dir = run_experiment(config)
	print(f"Saved part 1 results to: {run_dir.resolve()}")


if __name__ == "__main__":
	main()