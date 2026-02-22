import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataVariant:
    recipe: str
    invariant_fraction: float
    stage_proportions: str


@dataclass(frozen=True)
class ModelVariant:
    lr: float
    depth: int
    hidden_size: int
    curriculum: bool


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one string value")
    return values


def _parse_bool_list(raw: str) -> list[bool]:
    values: list[bool] = []
    for item in _parse_str_list(raw):
        lowered = item.lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            values.append(True)
        elif lowered in {"0", "false", "f", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"Cannot parse boolean value: {item}")
    return values


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _count_dataset_rows(dataset_dir: Path) -> int:
    total = 0
    for npz_file in sorted(dataset_dir.glob("*.npz")):
        with np.load(npz_file) as data:
            total += int(data["ox"].shape[0])
    return total


def _load_history_metrics(history_path: Path) -> dict[str, Any]:
    with open(history_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    history = payload.get("history", {})
    train_hist = history.get("train", [])
    val_hist = history.get("val", [])

    return {
        "final_train_loss": float(train_hist[-1]) if train_hist else None,
        "best_val_loss": min((float(v[1]) for v in val_hist), default=None),
        "num_train_points": len(train_hist),
        "num_val_points": len(val_hist),
    }


def _load_score_metrics(scorecard_path: Path) -> dict[str, Any]:
    with open(scorecard_path, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    per_case = [
        value
        for value in scorecard.values()
        if isinstance(value, dict) and "current_mae" in value and "current_mse" in value
    ]

    return {
        "evaluation_score": float(scorecard.get("Final_Score_Out_Of_100", 0.0)),
        "legacy_r2_score": float(scorecard.get("Legacy_R2_Score_Out_Of_100", 0.0)),
        "mean_current_mae": float(np.mean([m["current_mae"] for m in per_case])) if per_case else None,
        "mean_current_mse": float(np.mean([m["current_mse"] for m in per_case])) if per_case else None,
        "scorecard": scorecard,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Large-scale grid tuner across data recipes and surrogate hyperparameters"
    )

    parser.add_argument("--output-root", type=str, default="tune_runs", help="Output directory for all tuning artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    parser.add_argument("--n-samples-gen", type=int, default=64, help="Base simulations per generated dataset")
    parser.add_argument("--n-chunks", type=int, default=1, help="Chunk count per generated dataset")
    parser.add_argument("--max-species", type=int, default=5, help="Max species for generation")
    parser.add_argument("--nx", type=int, default=32, help="Spatial grid points for generation")
    parser.add_argument("--workers", type=int, default=1, help="Worker count for generation")
    parser.add_argument("--target-sig-len", type=int, default=200, help="Signal resample length")

    parser.add_argument(
        "--recipes",
        type=str,
        default="curriculum_multitask,stress_mixture",
        help="Comma-separated recipes for generation",
    )
    parser.add_argument(
        "--invariant-fractions",
        type=str,
        default="0.0,0.35",
        help="Comma-separated invariant fractions for generation",
    )
    parser.add_argument(
        "--stage-proportions",
        type=str,
        default="0.4,0.35,0.25",
        help="Stage proportions for curriculum recipe",
    )

    parser.add_argument("--epochs", type=int, default=120, help="Training epochs per run")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--n-samples-train", type=int, default=0, help="Samples to use in training (0 = all)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")

    parser.add_argument("--learning-rates", type=str, default="1e-3,5e-4", help="Comma-separated learning rates")
    parser.add_argument("--depths", type=str, default="3,4", help="Comma-separated network depths")
    parser.add_argument("--hidden-sizes", type=str, default="128,192", help="Comma-separated hidden sizes")
    parser.add_argument(
        "--curriculum-modes",
        type=str,
        default="true,false",
        help="Comma-separated booleans for --curriculum mode",
    )

    parser.add_argument("--top-k", type=int, default=5, help="How many top runs to print")
    parser.add_argument(
        "--write-config",
        type=str,
        default="config.json",
        help="Path to write best training config",
    )
    parser.add_argument(
        "--recommendation-path",
        type=str,
        default=None,
        help="Path for ranked recommendation JSON (default: <output-root>/recommended_config.json)",
    )

    return parser.parse_args()


def tune() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    dataset_root = output_root / "datasets"
    runs_root = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    recipes = _parse_str_list(args.recipes)
    invariant_fracs = _parse_float_list(args.invariant_fractions)
    lrs = _parse_float_list(args.learning_rates)
    depths = _parse_int_list(args.depths)
    hidden_sizes = _parse_int_list(args.hidden_sizes)
    curriculum_modes = _parse_bool_list(args.curriculum_modes)

    data_variants = [
        DataVariant(recipe=recipe, invariant_fraction=inv_frac, stage_proportions=args.stage_proportions)
        for recipe, inv_frac in itertools.product(recipes, invariant_fracs)
    ]
    model_variants = [
        ModelVariant(lr=lr, depth=depth, hidden_size=hidden, curriculum=curriculum)
        for lr, depth, hidden, curriculum in itertools.product(lrs, depths, hidden_sizes, curriculum_modes)
    ]

    print(f"Data variants: {len(data_variants)}")
    print(f"Model variants: {len(model_variants)}")
    print(f"Total runs: {len(data_variants) * len(model_variants)}")

    prepared_datasets: list[dict[str, Any]] = []
    for data_idx, data_variant in enumerate(data_variants):
        data_slug = (
            f"d{data_idx:02d}_{data_variant.recipe}"
            f"_inv{data_variant.invariant_fraction:.2f}".replace(".", "p")
        )
        dataset_dir = dataset_root / data_slug
        dataset_dir.mkdir(parents=True, exist_ok=True)

        scenario_seed = args.seed + 1000 * (data_idx + 1)

        gen_cmd = [
            sys.executable,
            "-m",
            "ecsfm.data.generate",
            "--n-samples",
            str(args.n_samples_gen),
            "--n-chunks",
            str(args.n_chunks),
            "--max-species",
            str(args.max_species),
            "--nx",
            str(args.nx),
            "--workers",
            str(args.workers),
            "--seed",
            str(scenario_seed),
            "--recipe",
            data_variant.recipe,
            "--stage-proportions",
            data_variant.stage_proportions,
            "--target-sig-len",
            str(args.target_sig_len),
            "--invariant-fraction",
            str(data_variant.invariant_fraction),
            "--output-dir",
            str(dataset_dir),
        ]
        if data_variant.invariant_fraction <= 0.0:
            gen_cmd.append("--no-invariants")

        print(f"\n=== Preparing dataset {data_slug} ===")
        _run(gen_cmd)
        rows = _count_dataset_rows(dataset_dir)

        prepared_datasets.append(
            {
                "data_idx": data_idx,
                "data_slug": data_slug,
                "dataset_dir": str(dataset_dir),
                "rows": rows,
                **asdict(data_variant),
            }
        )
        print(f"Prepared dataset {data_slug}: rows={rows}")

    results: list[dict[str, Any]] = []
    run_counter = 0

    for data_info in prepared_datasets:
        for model_idx, model_variant in enumerate(model_variants):
            run_dir = runs_root / f"run_{run_counter:03d}"
            artifact_dir = run_dir / "artifacts"
            eval_dir = run_dir / "eval"
            run_dir.mkdir(parents=True, exist_ok=True)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            run_seed = args.seed + 10_000 + run_counter

            train_cmd = [
                sys.executable,
                "-m",
                "ecsfm.fm.train",
                "--dataset",
                data_info["dataset_dir"],
                "--artifact-dir",
                str(artifact_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--n-samples",
                str(args.n_samples_train),
                "--val-split",
                str(args.val_split),
                "--lr",
                str(model_variant.lr),
                "--depth",
                str(model_variant.depth),
                "--hidden-size",
                str(model_variant.hidden_size),
                "--seed",
                str(run_seed),
                "--new-run",
            ]
            if model_variant.curriculum:
                train_cmd.append("--curriculum")
            else:
                train_cmd.append("--no-curriculum")

            eval_cmd = [
                sys.executable,
                "-m",
                "ecsfm.fm.eval_classical",
                "--checkpoint",
                str(artifact_dir / "surrogate_model.eqx"),
                "--output-dir",
                str(eval_dir),
                "--seed",
                str(run_seed),
            ]

            print(
                f"\n=== Run {run_counter + 1}/{len(prepared_datasets) * len(model_variants)} "
                f"(dataset={data_info['data_slug']}, model={model_idx}) ==="
            )

            t0 = time.perf_counter()
            result: dict[str, Any] = {
                "run_id": run_counter,
                "data": data_info,
                "model": asdict(model_variant),
                "seed": run_seed,
                "artifact_dir": str(artifact_dir),
                "eval_dir": str(eval_dir),
                "status": "pending",
            }

            try:
                _run(train_cmd)
                _run(eval_cmd)

                history_path = artifact_dir / "training_history.json"
                scorecard_path = eval_dir / "evaluation_scorecard.json"
                if not history_path.exists():
                    raise FileNotFoundError(f"Missing training history at {history_path}")
                if not scorecard_path.exists():
                    raise FileNotFoundError(f"Missing scorecard at {scorecard_path}")

                history_metrics = _load_history_metrics(history_path)
                score_metrics = _load_score_metrics(scorecard_path)

                result.update(history_metrics)
                result.update(score_metrics)
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = str(exc)
                print(f"Run failed: {exc}")

            elapsed_sec = time.perf_counter() - t0
            result["elapsed_sec"] = elapsed_sec
            results.append(result)

            if result["status"] == "ok":
                print(
                    f"Completed run {run_counter}: score={result['evaluation_score']:.2f}, "
                    f"legacy={result['legacy_r2_score']:.2f}, "
                    f"mean_mae={result['mean_current_mae']}, "
                    f"best_val={result['best_val_loss']}, elapsed={elapsed_sec:.1f}s"
                )

            run_counter += 1

    successful = [r for r in results if r["status"] == "ok"]
    ranked = sorted(
        successful,
        key=lambda r: (
            -float(r["evaluation_score"]),
            float(r["mean_current_mae"] if r["mean_current_mae"] is not None else np.inf),
            float(r["best_val_loss"] if r["best_val_loss"] is not None else np.inf),
        ),
    )

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    summary = {
        "args": vars(args),
        "data_variants": prepared_datasets,
        "model_variants": [asdict(m) for m in model_variants],
        "results": results,
        "ranked": ranked,
    }

    summary_path = output_root / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote summary: {summary_path}")

    if not ranked:
        print("No successful runs found. Skipping recommendation output.")
        return

    print("\n--- Top Runs ---")
    for item in ranked[: max(1, args.top_k)]:
        print(
            f"rank={item['rank']} run={item['run_id']} score={item['evaluation_score']:.2f} "
            f"mean_mae={item['mean_current_mae']} best_val={item['best_val_loss']} "
            f"recipe={item['data']['recipe']} inv_frac={item['data']['invariant_fraction']} "
            f"curriculum={item['model']['curriculum']} lr={item['model']['lr']} "
            f"depth={item['model']['depth']} hidden={item['model']['hidden_size']}"
        )

    best = ranked[0]

    best_train_config = {
        "data_path": best["data"]["dataset_dir"],
        "artifact_dir": best["artifact_dir"],
        "n_samples": int(args.n_samples_train),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(best["model"]["lr"]),
        "hidden_size": int(best["model"]["hidden_size"]),
        "depth": int(best["model"]["depth"]),
        "seed": int(best["seed"]),
        "new_run": False,
        "val_split": float(args.val_split),
        "curriculum": bool(best["model"]["curriculum"]),
    }

    write_config_path = Path(args.write_config)
    with open(write_config_path, "w", encoding="utf-8") as f:
        json.dump(best_train_config, f, indent=2)
    print(f"Wrote best train config to {write_config_path}")

    recommendation_payload = {
        "selected_run": {
            "run_id": best["run_id"],
            "rank": best["rank"],
            "evaluation_score": best["evaluation_score"],
            "legacy_r2_score": best["legacy_r2_score"],
            "mean_current_mae": best["mean_current_mae"],
            "mean_current_mse": best["mean_current_mse"],
            "best_val_loss": best["best_val_loss"],
            "final_train_loss": best["final_train_loss"],
            "artifact_dir": best["artifact_dir"],
            "eval_dir": best["eval_dir"],
        },
        "recommended_generation": {
            "recipe": best["data"]["recipe"],
            "invariant_fraction": best["data"]["invariant_fraction"],
            "stage_proportions": best["data"]["stage_proportions"],
            "n_samples_gen": args.n_samples_gen,
            "n_chunks": args.n_chunks,
            "max_species": args.max_species,
            "nx": args.nx,
            "workers": args.workers,
            "target_sig_len": args.target_sig_len,
        },
        "recommended_train_config": best_train_config,
        "top_runs": [
            {
                "rank": item["rank"],
                "run_id": item["run_id"],
                "evaluation_score": item["evaluation_score"],
                "mean_current_mae": item["mean_current_mae"],
                "best_val_loss": item["best_val_loss"],
                "recipe": item["data"]["recipe"],
                "invariant_fraction": item["data"]["invariant_fraction"],
                "curriculum": item["model"]["curriculum"],
                "lr": item["model"]["lr"],
                "depth": item["model"]["depth"],
                "hidden_size": item["model"]["hidden_size"],
            }
            for item in ranked[: max(1, args.top_k)]
        ],
    }

    recommendation_path = (
        Path(args.recommendation_path)
        if args.recommendation_path is not None
        else output_root / "recommended_config.json"
    )
    with open(recommendation_path, "w", encoding="utf-8") as f:
        json.dump(recommendation_payload, f, indent=2)
    print(f"Wrote recommendation bundle to {recommendation_path}")


if __name__ == "__main__":
    tune()
