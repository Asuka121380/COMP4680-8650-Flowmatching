from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scripts.part2_experiment import load_config, run_experiment


def load_sweep_config(config_path: Path) -> dict:
	with config_path.open("rb") as f:
		return tomllib.load(f)


def _load_result_arrays(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
	result_path = run_dir / "result.npz"
	if not result_path.exists():
		raise FileNotFoundError(f"Missing result file: {result_path}")
	data = np.load(result_path, allow_pickle=True)
	return data["real"], data["generated"]


def _make_summary_figure(
	group_key: tuple[int, str, str],
	group_runs: dict[str, Path],
	output_dir: Path,
	plot_cfg: dict,
) -> Path:
	dim, prediction_type, loss_type = group_key
	rows = int(plot_cfg.get("summary_rows", 2))
	cols = int(plot_cfg.get("summary_cols", 3))
	if rows != 2 or cols != 3:
		raise ValueError("This summary builder currently expects a 2x3 layout.")
	dataset_order = plot_cfg.get("summary_datasets", ["swiss_roll", "gaussians", "circles"])
	figure_title = plot_cfg.get("summary_title", f"Part 2 — pred={prediction_type}, loss={loss_type}, D={dim}")
	out_pattern = plot_cfg.get("summary_output_pattern", "d{dim}_{prediction_type}pred_{loss_type}loss_summary_2x3.png")
	out_name = out_pattern.format(dim=dim, prediction_type=prediction_type, loss_type=loss_type)
	figure_dir = output_dir / "summary"
	figure_dir.mkdir(parents=True, exist_ok=True)
	figure_path = figure_dir / out_name

	fig, axes = plt.subplots(rows, cols, figsize=(16, 9), sharex=True, sharey=True)
	for col, dataset_name in enumerate(dataset_order):
		run_dir = group_runs[dataset_name]
		real, generated = _load_result_arrays(run_dir)
		axes[0, col].scatter(real[:, 0], real[:, 1], s=5, alpha=0.7, linewidths=0)
		axes[0, col].set_title(f"{dataset_name} D={dim} — ground truth")
		axes[1, col].scatter(generated[:, 0], generated[:, 1], s=5, alpha=0.7, linewidths=0)
		axes[1, col].set_title(f"{dataset_name} D={dim} — generated")
		for row in range(rows):
			axes[row, col].set_aspect("equal", adjustable="box")
			axes[row, col].grid(alpha=0.25)
			axes[row, col].set_xlabel("x")
			axes[row, col].set_ylabel("y")
	fig.suptitle(figure_title)
	fig.tight_layout()
	fig.savefig(figure_path, dpi=int(plot_cfg.get("dpi", 200)))
	plt.close(fig)
	return figure_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Part 2 sweep over 36 experiments")
	parser.add_argument("--config", type=Path, default=Path("configs") / "part2_sweep.toml")
	parser.add_argument("--base-config", type=Path, default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--run-name-prefix", type=str, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	sweep_cfg = load_sweep_config(args.config)
	sweep = sweep_cfg.get("sweep", {})
	plot_cfg = sweep_cfg.get("plot", {})
	base_config_path = args.base_config or Path(sweep.get("base_config", "configs/part2_experiment.toml"))
	base_config = load_config(base_config_path)
	if args.output_dir is not None:
		base_config.output_dir = args.output_dir
	elif "output_dir" in sweep:
		base_config.output_dir = Path(sweep["output_dir"])
	datasets = sweep.get("datasets", ["swiss_roll", "gaussians", "circles"])
	dims = sweep.get("dims", [2, 8, 32])
	prediction_types = sweep.get("prediction_types", ["x", "v"])
	loss_types = sweep.get("loss_types", ["x", "v"])
	run_name_prefix = args.run_name_prefix
	if run_name_prefix is None:
		run_name_prefix = sweep.get("run_name_prefix", "")
	save_single_in_sweep = bool(plot_cfg.get("save_single_comparison_in_sweep", False))

	group_runs: dict[tuple[int, str, str], dict[str, Path]] = {}

	total = len(datasets) * len(dims) * len(prediction_types) * len(loss_types)
	index = 0
	for dim in dims:
		for prediction_type in prediction_types:
			for loss_type in loss_types:
				group_key = (int(dim), str(prediction_type), str(loss_type))
				group_runs[group_key] = {}
				for dataset_name in datasets:
					index += 1
					run_name = None
					if run_name_prefix:
						run_name = f"{run_name_prefix}_{dataset_name}_d{dim}_{prediction_type}{loss_type}"
					cfg = replace(
						base_config,
						dataset_name=dataset_name,
						dim=int(dim),
						prediction_type=str(prediction_type),
						loss_type=str(loss_type),
						run_name=run_name,
					)
					run_dir = run_experiment(cfg, save_single_comparison=save_single_in_sweep)
					print(f"[{index}/{total}] {dataset_name} d={dim} {prediction_type}-pred {loss_type}-loss -> {run_dir}")
					group_runs[group_key][dataset_name] = run_dir
				figure_path = _make_summary_figure(group_key, group_runs[group_key], base_config.output_dir, plot_cfg)
				print(f"    summary -> {figure_path}")


if __name__ == "__main__":
	main()
