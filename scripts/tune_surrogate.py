import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search tuner for surrogate model hyperparameters")
    parser.add_argument("--dataset", "--data-path", dest="data_path", type=str, default="/tmp/ecsfm/dataset_massive", help="Dataset path for training")
    parser.add_argument("--artifact-root", type=str, default="tune_runs", help="Directory to store per-run artifacts")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per tuning run")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples per tuning run")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--learning-rates", type=str, default="1e-3,5e-4", help="Comma-separated learning rates")
    parser.add_argument("--depths", type=str, default="3,4", help="Comma-separated depths")
    parser.add_argument("--hidden-sizes", type=str, default="64,128", help="Comma-separated hidden sizes")
    return parser.parse_args()


def tune() -> None:
    args = parse_args()

    learning_rates = _parse_float_list(args.learning_rates)
    depths = _parse_int_list(args.depths)
    hidden_sizes = _parse_int_list(args.hidden_sizes)

    if not learning_rates or not depths or not hidden_sizes:
        raise ValueError("Hyperparameter grids must all be non-empty")

    run_grid = [
        (lr, depth, hs)
        for lr in learning_rates
        for depth in depths
        for hs in hidden_sizes
    ]

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting grid search with {len(run_grid)} runs")

    results: list[dict] = []

    for run_idx, (lr, depth, hs) in enumerate(run_grid):
        run_dir = artifact_root / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_artifacts = run_dir / "artifacts"

        config_payload = {
            "lr": lr,
            "depth": depth,
            "hidden_size": hs,
            "epochs": args.epochs,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "new_run": True,
            "val_split": args.val_split,
        }
        config_path = run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

        print(
            f"\n--- Run {run_idx + 1}/{len(run_grid)} "
            f"(lr={lr}, depth={depth}, hidden_size={hs}) ---"
        )

        cmd = [
            sys.executable,
            "-m",
            "ecsfm.fm.train",
            "--dataset",
            args.data_path,
            "--artifact-dir",
            str(run_artifacts),
            "--n-samples",
            str(args.n_samples),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(lr),
            "--depth",
            str(depth),
            "--hidden-size",
            str(hs),
            "--val-split",
            str(args.val_split),
            "--new-run",
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Run failed: {exc}")
            results.append(
                {
                    "run": run_idx,
                    "config": config_payload,
                    "status": "failed",
                    "min_val_loss": None,
                }
            )
            continue

        history_path = run_artifacts / "training_history.json"
        if not history_path.exists():
            print("Run finished but no training history was produced")
            results.append(
                {
                    "run": run_idx,
                    "config": config_payload,
                    "status": "missing_history",
                    "min_val_loss": None,
                }
            )
            continue

        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        val_history = history_data.get("history", {}).get("val", [])
        if not val_history:
            print("Run finished but contains no validation points")
            min_val = None
            status = "no_val"
        else:
            min_val = min(float(item[1]) for item in val_history)
            status = "ok"
            print(f"Min validation loss: {min_val:.6f}")

        results.append(
            {
                "run": run_idx,
                "config": config_payload,
                "status": status,
                "min_val_loss": min_val,
                "artifact_dir": str(run_artifacts),
            }
        )

    print("\n--- Hyperparameter Evaluation ---")
    successful = [r for r in results if r["status"] == "ok" and r["min_val_loss"] is not None]

    if not successful:
        print("No successful runs with validation metrics.")
        return

    best = min(successful, key=lambda r: r["min_val_loss"])
    best_cfg = dict(best["config"])

    print(
        f"Best run: {best['run']} | min_val_loss={best['min_val_loss']:.6f} | "
        f"lr={best_cfg['lr']} depth={best_cfg['depth']} hidden={best_cfg['hidden_size']}"
    )

    final_config = {
        "lr": best_cfg["lr"],
        "depth": best_cfg["depth"],
        "hidden_size": best_cfg["hidden_size"],
        "epochs": 1000,
        "n_samples": 500,
        "batch_size": args.batch_size,
        "new_run": False,
        "val_split": args.val_split,
    }

    final_config_path = Path("config.json")
    with open(final_config_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)

    summary_path = artifact_root / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote winning config to {final_config_path}")
    print(f"Wrote full tuning summary to {summary_path}")


if __name__ == "__main__":
    tune()
