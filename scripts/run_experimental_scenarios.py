import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scenario:
    slug: str
    recipe: str
    curriculum: bool
    invariants: bool
    invariant_fraction: float
    description: str


SCENARIOS = [
    Scenario(
        slug="baseline_random",
        recipe="baseline_random",
        curriculum=False,
        invariants=False,
        invariant_fraction=0.0,
        description="Uniform random task mix without curriculum staging or invariant pairs.",
    ),
    Scenario(
        slug="curriculum_multitask",
        recipe="curriculum_multitask",
        curriculum=True,
        invariants=False,
        invariant_fraction=0.0,
        description="Stage-aware curriculum multitask data without invariant augmentation.",
    ),
    Scenario(
        slug="curriculum_multitask_invariants",
        recipe="curriculum_multitask",
        curriculum=True,
        invariants=True,
        invariant_fraction=0.5,
        description="Curriculum multitask data with invariant pair augmentations.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end curriculum/multitask surrogate experiments and summarize outcomes"
    )
    parser.add_argument("--output-root", type=str, default="/tmp/ecsfm/experiments", help="Directory for all experiment artifacts")
    parser.add_argument("--n-samples", type=int, default=64, help="Base simulation count per scenario")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs per scenario")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--max-species", type=int, default=5, help="Maximum species during data generation")
    parser.add_argument("--nx", type=int, default=40, help="Spatial grid size during data generation")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--workers", type=int, default=1, help="Worker count for simulation generation")
    parser.add_argument("--target-sig-len", type=int, default=200, help="Resampled signal length")
    parser.add_argument("--stage-proportions", type=str, default="0.4,0.35,0.25", help="Curriculum stage proportions")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_training_metrics(history_path: Path) -> dict[str, Any]:
    with open(history_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    history = payload.get("history", {})
    train_hist = history.get("train", [])
    val_hist = history.get("val", [])
    best_val = min((float(item[1]) for item in val_hist), default=None)

    return {
        "final_train_loss": float(train_hist[-1]) if train_hist else None,
        "best_val_loss": best_val,
        "num_train_points": len(train_hist),
        "num_val_points": len(val_hist),
    }


def _load_eval_metrics(scorecard_path: Path) -> dict[str, Any]:
    with open(scorecard_path, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    per_case = [
        v
        for v in scorecard.values()
        if isinstance(v, dict) and "current_mae" in v and "current_mse" in v
    ]
    mean_mae = float(np.mean([m["current_mae"] for m in per_case])) if per_case else None
    mean_mse = float(np.mean([m["current_mse"] for m in per_case])) if per_case else None

    return {
        "evaluation_score": float(scorecard.get("Final_Score_Out_Of_100", 0.0)),
        "mean_current_mae": mean_mae,
        "mean_current_mse": mean_mse,
        "scorecard": scorecard,
    }


def _count_rows(dataset_dir: Path) -> int:
    total = 0
    for npz_file in sorted(dataset_dir.glob("*.npz")):
        with np.load(npz_file) as data:
            total += int(data["ox"].shape[0])
    return total


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _build_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Curriculum Multitask Experiment Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Scenario | Recipe | Curriculum | Invariants | Rows | Final Train Loss | Best Val Loss | Eval Score (/100) | Mean Current MAE | Mean Current MSE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for item in results:
        lines.append(
            f"| {item['scenario']} | {item['recipe']} | {item['curriculum']} | {item['invariants']} | "
            f"{item['rows']} | {_format_float(item['final_train_loss'])} | "
            f"{_format_float(item['best_val_loss'])} | {item['evaluation_score']:.2f} | "
            f"{_format_float(item['mean_current_mae'])} | {_format_float(item['mean_current_mse'])} |"
        )

    by_score = sorted(
        results,
        key=lambda x: (
            -float(x["evaluation_score"]),
            float(x["mean_current_mae"] if x["mean_current_mae"] is not None else np.inf),
        ),
    )
    lines.extend(
        [
            "",
            "## Outcome Analysis",
            "",
            f"Top scenario by score/tie-breaker: `{by_score[0]['scenario']}` "
            f"({by_score[0]['evaluation_score']:.2f}/100, mean MAE={_format_float(by_score[0]['mean_current_mae'])}).",
        ]
    )

    baseline = next((r for r in results if r["scenario"] == "baseline_random"), None)
    curriculum = next((r for r in results if r["scenario"] == "curriculum_multitask"), None)
    invariants = next((r for r in results if r["scenario"] == "curriculum_multitask_invariants"), None)

    if baseline and curriculum:
        delta = curriculum["evaluation_score"] - baseline["evaluation_score"]
        lines.append(
            f"Curriculum vs baseline eval delta: {delta:+.2f} points."
        )
        if baseline["mean_current_mae"] is not None and curriculum["mean_current_mae"] is not None:
            lines.append(
                "Curriculum vs baseline mean current MAE delta: "
                f"{curriculum['mean_current_mae'] - baseline['mean_current_mae']:+.6f}."
            )
    if curriculum and invariants:
        delta = invariants["evaluation_score"] - curriculum["evaluation_score"]
        lines.append(
            f"Invariants vs curriculum-only eval delta: {delta:+.2f} points."
        )
        if curriculum["mean_current_mae"] is not None and invariants["mean_current_mae"] is not None:
            lines.append(
                "Invariants vs curriculum-only mean current MAE delta: "
                f"{invariants['mean_current_mae'] - curriculum['mean_current_mae']:+.6f}."
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for idx, scenario in enumerate(SCENARIOS):
        scenario_dir = output_root / scenario.slug
        dataset_dir = scenario_dir / "dataset"
        artifact_dir = scenario_dir / "artifacts"
        eval_dir = scenario_dir / "eval"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        scenario_seed = args.seed + idx * 101

        print(f"\n=== Scenario: {scenario.slug} ===")
        print(scenario.description)

        gen_cmd = [
            sys.executable,
            "-m",
            "ecsfm.data.generate",
            "--n-samples",
            str(args.n_samples),
            "--n-chunks",
            "1",
            "--max-species",
            str(args.max_species),
            "--nx",
            str(args.nx),
            "--workers",
            str(args.workers),
            "--seed",
            str(scenario_seed),
            "--recipe",
            scenario.recipe,
            "--stage-proportions",
            args.stage_proportions,
            "--target-sig-len",
            str(args.target_sig_len),
            "--invariant-fraction",
            str(scenario.invariant_fraction),
            "--output-dir",
            str(dataset_dir),
        ]
        if not scenario.invariants:
            gen_cmd.append("--no-invariants")

        train_cmd = [
            sys.executable,
            "-m",
            "ecsfm.fm.train",
            "--dataset",
            str(dataset_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(scenario_seed),
            "--new-run",
        ]
        if scenario.curriculum:
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
            str(scenario_seed),
        ]

        t0 = time.perf_counter()
        _run(gen_cmd)
        _run(train_cmd)
        _run(eval_cmd)
        elapsed_sec = time.perf_counter() - t0

        history_path = artifact_dir / "training_history.json"
        scorecard_path = eval_dir / "evaluation_scorecard.json"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing training history at {history_path}")
        if not scorecard_path.exists():
            raise FileNotFoundError(f"Missing evaluation scorecard at {scorecard_path}")

        training_metrics = _load_training_metrics(history_path)
        eval_metrics = _load_eval_metrics(scorecard_path)
        rows = _count_rows(dataset_dir)

        results.append(
            {
                "scenario": scenario.slug,
                "recipe": scenario.recipe,
                "curriculum": scenario.curriculum,
                "invariants": scenario.invariants,
                "invariant_fraction": scenario.invariant_fraction,
                "rows": rows,
                "elapsed_sec": elapsed_sec,
                **training_metrics,
                **eval_metrics,
            }
        )

        print(
            f"Completed {scenario.slug}: rows={rows}, "
            f"final_train={training_metrics['final_train_loss']}, "
            f"best_val={training_metrics['best_val_loss']}, "
            f"score={eval_metrics['evaluation_score']:.2f}, "
            f"elapsed={elapsed_sec:.1f}s"
        )

    summary_path = output_root / "scenario_summary.json"
    report_path = output_root / "scenario_report.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report = _build_report(results)
    report_path.write_text(report, encoding="utf-8")

    print(f"\nWrote scenario summary to {summary_path}")
    print(f"Wrote scenario report to {report_path}")


if __name__ == "__main__":
    main()
