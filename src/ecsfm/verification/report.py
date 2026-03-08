"""Report generation for verification results.

Generates per-case reports with overlaid traces and error metric tables,
plus a summary matrix of all cases.  Uses matplotlib for plots and outputs
Markdown + PNG artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ecsfm.verification.comparator import ComparisonMetrics
from ecsfm.verification.runner import SimResult
from ecsfm.verification.test_cases import TestCaseSpec

logger = logging.getLogger(__name__)


def _ensure_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        output_dir = Path("/tmp/ecsfm/verification")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _pass_fail(metrics: ComparisonMetrics | None, spec: TestCaseSpec) -> str:
    """Determine PASS/FAIL for a case based on its tolerance dict."""
    if metrics is None:
        return "SKIP"

    tol = spec.tolerance
    if not tol:
        # No tolerance defined, pass if errors are finite
        if np.isfinite(metrics.l2_norm) and metrics.l2_norm < 1.0:
            return "PASS"
        return "FAIL"

    # Check various tolerance keys
    if "cottrell_rel_error_max" in tol:
        if metrics.max_relative_error > tol["cottrell_rel_error_max"]:
            return "FAIL"
    if "delta_ep_abs_tol" in tol:
        if metrics.l2_norm > tol["delta_ep_abs_tol"]:
            return "FAIL"
    if "comsol_l2_norm" in tol:
        if metrics.l2_norm > tol["comsol_l2_norm"]:
            return "FAIL"
    if "delta_ep_min_check" in tol:
        # peak_position_error stores actual delta_ep for this comparator
        if metrics.peak_position_error is not None:
            if metrics.peak_position_error < tol["delta_ep_min_check"]:
                return "FAIL"

    return "PASS"


def generate_case_report(
    spec: TestCaseSpec,
    sim_result: SimResult,
    ref_result: dict[str, np.ndarray] | None = None,
    metrics: ComparisonMetrics | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a per-case report with overlaid traces and error metrics.

    Returns the path to the generated Markdown report file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = _ensure_output_dir(output_dir)
    case_dir = output_dir / spec.name
    case_dir.mkdir(parents=True, exist_ok=True)

    status = _pass_fail(metrics, spec)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if spec.waveform.type == "eis":
        _plot_eis(axes, spec, sim_result, ref_result)
    elif spec.waveform.type == "cv":
        _plot_cv(axes, spec, sim_result, ref_result)
    else:
        _plot_step(axes, spec, sim_result, ref_result)

    fig.suptitle(f"{spec.name} [{status}]", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = case_dir / f"{spec.name}_traces.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # ---- Markdown ----
    lines = [
        f"# Verification: {spec.name}",
        "",
        f"**Status:** {status}",
        "",
        f"**Category:** {spec.category}",
        "",
        f"**Description:** {spec.description}",
        "",
        "## Error Metrics",
        "",
    ]

    if metrics is not None:
        lines.extend([
            "| Metric | Value |",
            "|---|---|",
            f"| L2 Norm | {metrics.l2_norm:.6e} |",
            f"| L-inf Norm | {metrics.l_inf_norm:.6e} |",
            f"| Max Relative Error | {metrics.max_relative_error:.6e} |",
        ])
        if metrics.peak_position_error is not None:
            lines.append(f"| Peak Position Error | {metrics.peak_position_error:.6e} |")
        if metrics.peak_current_error is not None:
            lines.append(f"| Peak Current Error | {metrics.peak_current_error:.6e} |")
    else:
        lines.append("_No analytical reference available for this case._")

    lines.extend([
        "",
        "## Traces",
        "",
        f"![Traces]({spec.name}_traces.png)",
        "",
    ])

    report_path = case_dir / f"{spec.name}_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ---- JSON metrics ----
    metrics_dict: dict[str, Any] = {
        "case_name": spec.name,
        "status": status,
        "category": spec.category,
    }
    if metrics is not None:
        metrics_dict["l2_norm"] = metrics.l2_norm
        metrics_dict["l_inf_norm"] = metrics.l_inf_norm
        metrics_dict["max_relative_error"] = metrics.max_relative_error
        metrics_dict["peak_position_error"] = metrics.peak_position_error
        metrics_dict["peak_current_error"] = metrics.peak_current_error

    json_path = case_dir / f"{spec.name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, default=str)

    logger.info("Generated report for '%s' at %s", spec.name, report_path)
    return report_path


def _plot_step(axes, spec, sim_result, ref_result):
    """Plot potential step traces."""
    ax_e, ax_i = axes

    ax_e.plot(sim_result.time, sim_result.potential, label="Sim E(t)")
    ax_e.set_xlabel("Time (s)")
    ax_e.set_ylabel("Potential (V)")
    ax_e.set_title("Applied Potential")
    ax_e.legend()
    ax_e.grid(True, alpha=0.3)

    ax_i.plot(sim_result.time, sim_result.current, label="Sim I(t)")
    if ref_result is not None:
        ax_i.plot(ref_result["time"], ref_result["current"], "--", label="Reference")
    ax_i.set_xlabel("Time (s)")
    ax_i.set_ylabel("Current (mA)")
    ax_i.set_title("Current Response")
    ax_i.legend()
    ax_i.grid(True, alpha=0.3)


def _plot_cv(axes, spec, sim_result, ref_result):
    """Plot cyclic voltammogram."""
    ax_cv, ax_it = axes

    ax_cv.plot(sim_result.potential, sim_result.current, label="Sim")
    if ref_result is not None:
        ax_cv.plot(ref_result["potential"], ref_result["current"], "--", label="Reference")
    ax_cv.set_xlabel("Potential (V)")
    ax_cv.set_ylabel("Current (mA)")
    ax_cv.set_title("Cyclic Voltammogram")
    ax_cv.legend()
    ax_cv.grid(True, alpha=0.3)

    ax_it.plot(sim_result.time, sim_result.current, label="Sim I(t)")
    if ref_result is not None:
        ax_it.plot(ref_result["time"], ref_result["current"], "--", label="Reference")
    ax_it.set_xlabel("Time (s)")
    ax_it.set_ylabel("Current (mA)")
    ax_it.set_title("Current vs Time")
    ax_it.legend()
    ax_it.grid(True, alpha=0.3)


def _plot_eis(axes, spec, sim_result, ref_result):
    """Plot EIS Nyquist and Bode diagrams."""
    ax_nyq, ax_bode = axes

    z_real = sim_result.metadata.get("z_real_ohm", sim_result.potential)
    z_imag = sim_result.metadata.get("z_imag_ohm", sim_result.current)
    freqs = sim_result.metadata.get("frequencies_hz", sim_result.time)

    ax_nyq.plot(z_real, -z_imag, "o-", label="Sim")
    ax_nyq.set_xlabel("Z' (Ohm)")
    ax_nyq.set_ylabel("-Z'' (Ohm)")
    ax_nyq.set_title("Nyquist Plot")
    ax_nyq.set_aspect("equal", adjustable="datalim")
    ax_nyq.legend()
    ax_nyq.grid(True, alpha=0.3)

    z_mag = sim_result.metadata.get("z_mag_ohm", np.sqrt(z_real**2 + z_imag**2))
    ax_bode.semilogx(freqs, z_mag, "o-", label="|Z|")
    ax_bode.set_xlabel("Frequency (Hz)")
    ax_bode.set_ylabel("|Z| (Ohm)")
    ax_bode.set_title("Bode Magnitude")
    ax_bode.legend()
    ax_bode.grid(True, alpha=0.3)


def generate_summary_report(
    results: dict[str, dict],
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a summary matrix of all verification cases.

    Parameters
    ----------
    results : dict
        Mapping of case_name -> {"spec": TestCaseSpec, "sim": SimResult,
        "metrics": ComparisonMetrics | None, "status": str}.
    output_dir : path, optional
        Where to write the report.

    Returns the path to the summary Markdown file.
    """
    output_dir = _ensure_output_dir(output_dir)

    n_pass = sum(1 for r in results.values() if r.get("status") == "PASS")
    n_fail = sum(1 for r in results.values() if r.get("status") == "FAIL")
    n_skip = sum(1 for r in results.values() if r.get("status") == "SKIP")
    n_total = len(results)
    overall = "PASS" if n_fail == 0 and n_pass > 0 else "FAIL"

    lines = [
        "# ECSFM Verification Summary",
        "",
        f"**Overall: {overall}** ({n_pass} pass, {n_fail} fail, {n_skip} skip out of {n_total})",
        "",
        "## Results Matrix",
        "",
        "| Case | Category | Status | L2 Norm | Max Rel Error |",
        "|---|---|---|---|---|",
    ]

    for name, info in sorted(results.items()):
        spec = info.get("spec")
        metrics = info.get("metrics")
        status = info.get("status", "SKIP")
        category = spec.category if spec else "?"

        if metrics is not None:
            l2 = f"{metrics.l2_norm:.4e}"
            max_rel = f"{metrics.max_relative_error:.4e}"
        else:
            l2 = "-"
            max_rel = "-"

        lines.append(f"| `{name}` | {category} | {status} | {l2} | {max_rel} |")

    lines.extend(["", ""])

    summary_path = output_dir / "verification_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # JSON summary
    json_data: dict[str, Any] = {
        "overall": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "cases": {},
    }
    for name, info in results.items():
        metrics = info.get("metrics")
        json_data["cases"][name] = {
            "status": info.get("status", "SKIP"),
            "l2_norm": metrics.l2_norm if metrics else None,
            "max_relative_error": metrics.max_relative_error if metrics else None,
        }

    json_path = output_dir / "verification_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)

    logger.info("Generated verification summary at %s", summary_path)
    return summary_path
