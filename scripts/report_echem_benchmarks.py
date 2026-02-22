from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecsfm.sim.benchmarks import run_canonical_benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical electrochemistry benchmarks and write an audit report."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/ecsfm/echem_benchmarks",
        help="Directory for benchmark report artifacts.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing PNG plots.",
    )
    return parser.parse_args()


def _build_markdown(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    rows = []
    for key in sorted(checks.keys()):
        status = "PASS" if checks[key] else "FAIL"
        rows.append(f"| `{key}` | {status} |")

    cv_k0 = summary["cv_vs_k0"]
    cv_scan = summary["cv_vs_scan_rate"]
    cv_conc = summary["cv_vs_concentration"]
    cottrell = summary["cottrell"]
    sensor = summary["sensor_impedance"]

    lines = [
        "# Canonical Electrochemistry Benchmark Report",
        "",
        f"Overall: {'PASS' if summary['overall_pass'] else 'FAIL'}",
        "",
        "## Check Matrix",
        "",
        "| Check | Status |",
        "|---|---|",
        *rows,
        "",
        "## Key Metrics",
        "",
        f"- CV deltaEp at low k0: {cv_k0['delta_ep'][0]:.6f} V",
        f"- CV deltaEp at high k0: {cv_k0['delta_ep'][-1]:.6f} V",
        f"- CV deltaEp scan gain: {cv_scan['delta_ep'][-1] - cv_scan['delta_ep'][0]:.6f} V",
        f"- CV ipc vs concentration fit R2: {cv_conc['fit']['r2']:.6f}",
        f"- CV ipc vs concentration fit slope: {cv_conc['fit']['slope']:.6f}",
        f"- Cottrell max relative error: {max(cottrell['rel_error']):.6f}",
        f"- Cottrell log-log slope: {cottrell['loglog_fit']['slope']:.6f}",
        f"- Sensor max amplitude rel error: {max(sensor['amp_rel_error']):.6f}",
        f"- Sensor max absolute phase error: {max(abs(v) for v in sensor['phase_error_rad']):.6f} rad",
        "",
    ]
    return "\n".join(lines)


def _write_plots(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: list[Path] = []

    cv_k0 = summary["cv_vs_k0"]
    cv_scan = summary["cv_vs_scan_rate"]
    cv_conc = summary["cv_vs_concentration"]
    cottrell = summary["cottrell"]
    sensor = summary["sensor_impedance"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.semilogx(cv_k0["k0_values"], cv_k0["delta_ep"], marker="o")
    ax.set_xlabel("k0 (cm/s)")
    ax.set_ylabel("Delta Ep (V)")
    ax.set_title("CV Peak Separation vs k0")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogx(cv_scan["scan_rates"], cv_scan["delta_ep"], marker="o")
    ax.set_xlabel("Scan Rate (V/s)")
    ax.set_ylabel("Delta Ep (V)")
    ax.set_title("CV Peak Separation vs Scan Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    x = np.asarray(cv_conc["concentrations"], dtype=float)
    y = np.asarray(cv_conc["ip_c"], dtype=float)
    slope = float(cv_conc["fit"]["slope"])
    intercept = float(cv_conc["fit"]["intercept"])
    ax.plot(x, y, "o", label="simulated")
    ax.plot(x, slope * x + intercept, "-", label="linear fit")
    ax.set_xlabel("Bulk Concentration (mM)")
    ax.set_ylabel("Cathodic Peak Current (mA)")
    ax.set_title("CV Peak Current vs Concentration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    t = np.asarray(cottrell["times_s"], dtype=float)
    flux = np.asarray(cottrell["flux"], dtype=float)
    flux_ref = np.asarray(cottrell["analytic_flux"], dtype=float)
    ax.loglog(t, flux, "o-", label="simulated")
    ax.loglog(t, flux_ref, "s--", label="Cottrell")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flux (mol cm^-2 s^-1)")
    ax.set_title("Cottrell Flux Benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    core_plot = output_dir / "canonical_cv_cottrell.png"
    fig.savefig(core_plot, dpi=160)
    plt.close(fig)
    plot_paths.append(core_plot)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    freq = np.asarray(sensor["freq_hz"], dtype=float)

    ax = axes[0]
    ax.semilogx(freq, sensor["amp_measured_mA"], "o-", label="measured")
    ax.semilogx(freq, sensor["amp_theory_mA"], "s--", label="theory")
    ax.set_ylabel("Current Amplitude (mA)")
    ax.set_title("Sensor Impedance Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.semilogx(freq, sensor["phase_measured_rad"], "o-", label="measured")
    ax.semilogx(freq, sensor["phase_theory_rad"], "s--", label="theory")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Sensor Impedance Phase")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    sensor_plot = output_dir / "sensor_bode.png"
    fig.savefig(sensor_plot, dpi=160)
    plt.close(fig)
    plot_paths.append(sensor_plot)

    return plot_paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_canonical_benchmarks()
    summary_path = output_dir / "benchmark_summary.json"
    report_path = output_dir / "benchmark_report.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    report_path.write_text(_build_markdown(summary), encoding="utf-8")

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = _write_plots(summary, output_dir)

    payload = {
        "overall_pass": bool(summary["overall_pass"]),
        "checks": summary["checks"],
        "artifacts": {
            "summary_json": str(summary_path),
            "report_md": str(report_path),
            "plots": [str(path) for path in plot_paths],
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
