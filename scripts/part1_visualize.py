from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure src/ is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import AVAILABLE_DATASETS, ToyDiffusionDataset


def _sample_points(x: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
	if max_points is None or len(x) <= max_points:
		return x
	rng = np.random.default_rng(seed)
	idx = rng.choice(len(x), size=max_points, replace=False)
	return x[idx]


def _plot_and_save(points: np.ndarray, title: str, save_path: Path, dpi: int) -> None:
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.7, linewidths=0)
	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(save_path, dpi=dpi)
	plt.close(fig)


def generate_part1_figures(output_dir: Path, data_dir: Path | None, max_points: int | None, dpi: int) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	for name in AVAILABLE_DATASETS:
		ds_2d = ToyDiffusionDataset(name=name, dim=2, data_dir=data_dir)
		ds_32d = ToyDiffusionDataset(name=name, dim=32, data_dir=data_dir)

		x_2d = ds_2d.data.detach().cpu().numpy()
		x_32d = ds_32d.data.detach().cpu().numpy()
		x_32_to_2d = ds_32d.to_2d(x_32d)

		x_2d = _sample_points(x_2d, max_points=max_points, seed=42)
		x_32_to_2d = _sample_points(x_32_to_2d, max_points=max_points, seed=42)

		_plot_and_save(
			points=x_2d,
			title=f"{name}: original 2D",
			save_path=output_dir / f"{name}_2d.png",
			dpi=dpi,
		)
		_plot_and_save(
			points=x_32_to_2d,
			title=f"{name}: 32D back-projected to 2D",
			save_path=output_dir / f"{name}_32d_to_2d.png",
			dpi=dpi,
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Part 3.1 Data Visualization")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("outputs") / "part1",
		help="Directory to save the six output figures.",
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=None,
		help="Optional path to data directory. Defaults to project_root/data.",
	)
	parser.add_argument(
		"--max-points",
		type=int,
		default=None,
		help="Optional cap on plotted points per figure.",
	)
	parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	generate_part1_figures(
		output_dir=args.output_dir,
		data_dir=args.data_dir,
		max_points=args.max_points,
		dpi=args.dpi,
	)
	print(f"Saved figures to: {args.output_dir.resolve()}")


if __name__ == "__main__":
	main()
