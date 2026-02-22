from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecsfm.analysis.evidence import (
    cv_trace_sweep_by_k0,
    cv_trace_sweep_by_scan_rate,
    dataset_recipe_audit,
    simulator_convergence_study,
)
from ecsfm.sim.benchmarks import run_canonical_benchmarks
from ecsfm.sim.multiphysics_benchmarks import run_multiphysics_benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reviewer-facing evidence package for simulator correctness and dataset generation."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/ecsfm/nature_evidence",
        help="Directory where evidence artifacts will be written.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed for scenario audits.")
    parser.add_argument(
        "--dataset-audit-samples",
        type=int,
        default=512,
        help="Base n_samples per recipe for dataset scenario audit.",
    )
    parser.add_argument(
        "--dataset-audit-nx",
        type=int,
        default=24,
        help="Spatial nx used in dataset scenario audit generation.",
    )
    parser.add_argument(
        "--max-species",
        type=int,
        default=5,
        help="Maximum species used in scenario audit generation.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_plots(
    output_dir: Path,
    cv_k0: dict[str, Any],
    cv_scan: dict[str, Any],
    convergence: dict[str, Any],
    dataset_audit: dict[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for row in cv_k0["traces"]:
        ax.plot(row["E_hist"], row["I_hist"], lw=1.3, label=f"k0={row['k0']:.0e}")
    ax.set_title("Canonical CV Overlay (k0 Sweep)")
    ax.set_xlabel("E (V)")
    ax.set_ylabel("I (mA)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for row in cv_scan["traces"]:
        ax.plot(row["E_hist"], row["I_hist"], lw=1.3, label=f"v={row['scan_rate']:.2g} V/s")
    ax.set_title("Canonical CV Overlay (Scan Sweep)")
    ax.set_xlabel("E (V)")
    ax.set_ylabel("I (mA)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    cv_overlay = output_dir / "canonical_cv_overlays.png"
    fig.savefig(cv_overlay, dpi=170)
    plt.close(fig)
    artifacts.append(str(cv_overlay))

    rows = convergence["rows"]
    nxs = np.asarray([int(row["nx"]) for row in rows], dtype=int)
    nrmse = np.asarray([float(row["nrmse_vs_ref"]) for row in rows], dtype=float)
    dep = np.asarray([float(row["delta_ep_abs_error_vs_ref"]) for row in rows], dtype=float)
    dpeak = np.asarray([float(row["peak_abs_rel_error_vs_ref"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(nxs, nrmse, "o-")
    axes[0].set_title("Current NRMSE vs nx")
    axes[0].set_xlabel("nx")
    axes[0].set_ylabel("NRMSE")
    axes[0].grid(alpha=0.3)

    axes[1].plot(nxs, dep, "o-")
    axes[1].set_title("DeltaEp Abs Error vs nx")
    axes[1].set_xlabel("nx")
    axes[1].set_ylabel("|DeltaEp - Ref| (V)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(nxs, dpeak, "o-")
    axes[2].set_title("Peak Current Rel Error vs nx")
    axes[2].set_xlabel("nx")
    axes[2].set_ylabel("Relative Error")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    convergence_plot = output_dir / "simulator_convergence.png"
    fig.savefig(convergence_plot, dpi=170)
    plt.close(fig)
    artifacts.append(str(convergence_plot))

    recipes = [row["recipe"] for row in dataset_audit["rows"]]
    x = np.arange(len(recipes))
    width = 0.26
    task_entropy = np.asarray([float(row["task_entropy"]) for row in dataset_audit["rows"]], dtype=float)
    stage_entropy = np.asarray([float(row["stage_entropy"]) for row in dataset_audit["rows"]], dtype=float)
    aug_frac = np.asarray([float(row["augmentation_fraction"]) for row in dataset_audit["rows"]], dtype=float)
    permute_ok = np.asarray([float(row["permute_pass_fraction"]) for row in dataset_audit["rows"]], dtype=float)
    scale_ok = np.asarray([float(row["scale_within_20pct_fraction"]) for row in dataset_audit["rows"]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.bar(x - width, task_entropy, width, label="task entropy")
    ax.bar(x, stage_entropy, width, label="stage entropy")
    ax.bar(x + width, aug_frac, width, label="augmentation fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(recipes, rotation=20, ha="right")
    ax.set_title("Dataset Scenario Diversity")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - width / 2, permute_ok, width, label="permute invariant pass")
    ax.bar(x + width / 2, scale_ok, width, label="scale invariant pass")
    ax.set_xticks(x)
    ax.set_xticklabels(recipes, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Invariant Consistency Checks")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    dataset_plot = output_dir / "dataset_scenario_audit.png"
    fig.savefig(dataset_plot, dpi=170)
    plt.close(fig)
    artifacts.append(str(dataset_plot))
    return artifacts


def _build_markdown_report(
    canonical: dict[str, Any],
    multiphysics: dict[str, Any],
    convergence: dict[str, Any],
    dataset_audit: dict[str, Any],
) -> str:
    lines = [
        "# Nature-Style Simulator Evidence Report",
        "",
        f"Canonical benchmark overall pass: **{canonical['overall_pass']}**",
        "",
        "## Canonical Benchmarks",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for key, value in sorted(canonical["checks"].items()):
        lines.append(f"| `{key}` | {'PASS' if value else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## Multiphysics Fouling/EIS Benchmarks",
            "",
            f"Multiphysics benchmark overall pass: **{multiphysics['overall_pass']}**",
            "",
            "| Check | Status |",
            "|---|---|",
        ]
    )
    for key, value in sorted(multiphysics["checks"].items()):
        lines.append(f"| `{key}` | {'PASS' if value else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## Convergence Study",
            "",
            "| nx | Current NRMSE vs ref | |DeltaEp - Ref| (V) | Peak Current Rel Error |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in convergence["rows"]:
        lines.append(
            f"| {row['nx']} | {row['nrmse_vs_ref']:.6f} | "
            f"{row['delta_ep_abs_error_vs_ref']:.6f} | {row['peak_abs_rel_error_vs_ref']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Scenario Audit",
            "",
            "| Recipe | Rows | Aug. Fraction | Task Entropy | Stage Entropy | Permute Pass | Scale Pass (<20%) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_audit["rows"]:
        lines.append(
            f"| {row['recipe']} | {row['total_rows']} | {row['augmentation_fraction']:.3f} | "
            f"{row['task_entropy']:.3f} | {row['stage_entropy']:.3f} | "
            f"{row['permute_pass_fraction']:.3f} | {row['scale_within_20pct_fraction']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Canonical checks validate physically expected trends in CV peak separation, concentration scaling, Cottrell decay, and RC impedance response.",
            "- Convergence table quantifies numerical stabilization as `nx` increases against a fixed high-resolution reference.",
            "- Dataset audit measures coverage diversity across recipes and verifies invariance augmentations behave as designed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical = run_canonical_benchmarks()
    multiphysics = run_multiphysics_benchmarks()
    convergence = simulator_convergence_study(np.array([20, 28, 36, 44, 56], dtype=int), reference_nx=72)
    dataset_audit = dataset_recipe_audit(
        n_samples=args.dataset_audit_samples,
        max_species=args.max_species,
        nx=args.dataset_audit_nx,
        seed=args.seed,
    )
    cv_k0 = cv_trace_sweep_by_k0(np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=float))
    cv_scan = cv_trace_sweep_by_scan_rate(np.array([0.5, 1.0, 2.0, 4.0], dtype=float))

    _write_json(output_dir / "canonical_benchmarks.json", canonical)
    _write_json(output_dir / "multiphysics_benchmarks.json", multiphysics)
    _write_json(output_dir / "convergence_study.json", convergence)
    _write_json(output_dir / "dataset_recipe_audit.json", dataset_audit)
    _write_json(
        output_dir / "cv_trace_sweeps.json",
        {
            "k0_summary": cv_k0["summary"],
            "scan_summary": cv_scan["summary"],
        },
    )

    report_md = _build_markdown_report(canonical, multiphysics, convergence, dataset_audit)
    report_path = output_dir / "nature_evidence_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    plot_paths: list[str] = []
    if not args.no_plots:
        plot_paths = _write_plots(output_dir, cv_k0, cv_scan, convergence, dataset_audit)

    payload = {
        "output_dir": str(output_dir),
        "overall_pass": bool(canonical["overall_pass"] and multiphysics["overall_pass"]),
        "recipe_ranking": dataset_audit["recipe_ranking"],
        "artifacts": {
            "canonical_benchmarks": str(output_dir / "canonical_benchmarks.json"),
            "multiphysics_benchmarks": str(output_dir / "multiphysics_benchmarks.json"),
            "convergence_study": str(output_dir / "convergence_study.json"),
            "dataset_recipe_audit": str(output_dir / "dataset_recipe_audit.json"),
            "cv_trace_sweeps": str(output_dir / "cv_trace_sweeps.json"),
            "report_md": str(report_path),
            "plots": plot_paths,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
