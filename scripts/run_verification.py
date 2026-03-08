#!/usr/bin/env python
"""Run the ECSFM verification suite.

Usage:
    python scripts/run_verification.py                        # all cases
    python scripts/run_verification.py --cases cottrell_step   # single case
    python scripts/run_verification.py --comsol               # include COMSOL
    python scripts/run_verification.py --offline               # use cached data
    python scripts/run_verification.py --export-comsol-data    # export COMSOL CSVs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from ecsfm.verification.comparator import compare_against_analytical, compare_traces
from ecsfm.verification.comsol_runner import (
    export_reference_data,
    is_comsol_available,
    load_reference_data,
    run_comsol_case,
)
from ecsfm.verification.report import generate_case_report, generate_summary_report
from ecsfm.verification.runner import run_case
from ecsfm.verification.test_cases import CANONICAL_CASES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ECSFM verification suite against analytical and COMSOL references."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["all"],
        help="Case names to run, or 'all' for everything. Default: all.",
    )
    parser.add_argument(
        "--comsol",
        action="store_true",
        help="Include live COMSOL comparison (requires MPh + COMSOL license).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached COMSOL reference data instead of running COMSOL live.",
    )
    parser.add_argument(
        "--export-comsol-data",
        action="store_true",
        help="Run COMSOL for all cases and export CSV reference data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/ecsfm/verification",
        help="Directory for report artifacts.",
    )
    parser.add_argument(
        "--comsol-data-dir",
        type=str,
        default=None,
        help="Directory containing cached COMSOL CSV files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which cases to run
    if "all" in args.cases:
        case_names = list(CANONICAL_CASES.keys())
    else:
        case_names = []
        for name in args.cases:
            if name not in CANONICAL_CASES:
                logger.error("Unknown case: '%s'. Available: %s", name, list(CANONICAL_CASES.keys()))
                sys.exit(1)
            case_names.append(name)

    # Handle export-only mode
    if args.export_comsol_data:
        if not is_comsol_available():
            logger.error("COMSOL is not available. Cannot export reference data.")
            sys.exit(1)
        comsol_dir = Path(args.comsol_data_dir) if args.comsol_data_dir else (
            Path(__file__).resolve().parents[1] / "data" / "comsol_reference"
        )
        for name in case_names:
            spec = CANONICAL_CASES[name]
            logger.info("Exporting COMSOL data for '%s'...", name)
            path = export_reference_data(spec, comsol_dir)
            if path:
                logger.info("  -> %s", path)
            else:
                logger.warning("  -> export failed for '%s'", name)
        return

    # Run verification
    use_comsol_live = args.comsol and is_comsol_available()
    if args.comsol and not is_comsol_available():
        logger.warning("--comsol requested but COMSOL is not available. Falling back to offline/analytical.")

    all_results: dict[str, dict] = {}

    for name in case_names:
        spec = CANONICAL_CASES[name]
        logger.info("Running case: '%s' (%s)...", name, spec.category)

        try:
            sim_result = run_case(spec)
        except Exception as exc:
            logger.error("  Simulation FAILED for '%s': %s", name, exc)
            all_results[name] = {
                "spec": spec,
                "sim": None,
                "metrics": None,
                "status": "FAIL",
            }
            continue

        # Compare against analytical solutions
        analytical_metrics = compare_against_analytical(spec, sim_result)

        # Try COMSOL comparison
        ref_result = None
        comsol_metrics = None

        if use_comsol_live:
            comsol_data = run_comsol_case(spec)
            if comsol_data is not None:
                ref_result = comsol_data
                comsol_metrics = compare_traces(
                    sim_result.time, sim_result.current,
                    comsol_data["time"], comsol_data["current"],
                )
        elif args.offline:
            cached = load_reference_data(name, data_dir=args.comsol_data_dir)
            if cached is not None:
                ref_result = cached
                comsol_metrics = compare_traces(
                    sim_result.time, sim_result.current,
                    cached["time"], cached["current"],
                )

        # Use the best available metrics
        metrics = analytical_metrics or comsol_metrics

        # Determine pass/fail
        from ecsfm.verification.report import _pass_fail
        status = _pass_fail(metrics, spec)

        # Generate per-case report
        if not args.no_plots:
            try:
                generate_case_report(
                    spec, sim_result, ref_result=ref_result, metrics=metrics,
                    output_dir=output_dir,
                )
            except Exception as exc:
                logger.warning("  Report generation failed for '%s': %s", name, exc)

        all_results[name] = {
            "spec": spec,
            "sim": sim_result,
            "metrics": metrics,
            "status": status,
        }

        logger.info("  -> %s (L2=%.4e)", status, metrics.l2_norm if metrics else float("nan"))

    # Generate summary
    summary_path = generate_summary_report(all_results, output_dir=output_dir)

    # Print summary to stdout
    n_pass = sum(1 for r in all_results.values() if r["status"] == "PASS")
    n_fail = sum(1 for r in all_results.values() if r["status"] == "FAIL")
    n_skip = sum(1 for r in all_results.values() if r["status"] == "SKIP")

    print("\n" + "=" * 60)
    print("ECSFM Verification Results")
    print("=" * 60)
    for name, info in sorted(all_results.items()):
        metrics = info["metrics"]
        l2_str = f"L2={metrics.l2_norm:.4e}" if metrics else "N/A"
        print(f"  {info['status']:4s}  {name:30s}  {l2_str}")
    print("-" * 60)
    print(f"Total: {len(all_results)} | Pass: {n_pass} | Fail: {n_fail} | Skip: {n_skip}")
    overall = "PASS" if n_fail == 0 and n_pass > 0 else "FAIL"
    print(f"Overall: {overall}")
    print(f"Report: {summary_path}")
    print("=" * 60)

    # Exit with non-zero if any failures
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
