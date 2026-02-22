from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecsfm.sim.multiphysics_benchmarks import (
    multiphysics_impedance_sweep,
    randles_misfit_benchmark,
    run_multiphysics_benchmarks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiphysics electrochem benchmark report")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/ecsfm/multiphysics_benchmarks",
        help="Directory to write benchmark report artifacts",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def _plot_impedance_comparison(output_dir: Path, frequencies_hz: np.ndarray) -> list[str]:
    clean = multiphysics_impedance_sweep(frequencies_hz, initial_theta=0.0)
    fouled = multiphysics_impedance_sweep(frequencies_hz, initial_theta=0.8)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.loglog(clean["freq_hz"], clean["z_mag_ohm"], "o-", label="clean")
    ax.loglog(fouled["freq_hz"], fouled["z_mag_ohm"], "o-", label="fouled")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Z| (ohm)")
    ax.set_title("Bode Magnitude")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    ax = axes[1]
    ax.semilogx(clean["freq_hz"], clean["z_phase_rad"], "o-", label="clean")
    ax.semilogx(fouled["freq_hz"], fouled["z_phase_rad"], "o-", label="fouled")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Bode Phase")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig.tight_layout()
    bode_path = output_dir / "multiphysics_bode.png"
    fig.savefig(bode_path, dpi=180)
    plt.close(fig)

    misfit = randles_misfit_benchmark(initial_theta=0.8, frequencies_hz=frequencies_hz)
    fit = misfit["fit"]
    freq = np.asarray(misfit["freq_hz"], dtype=float)
    z_data = np.asarray(misfit["z_real_ohm"], dtype=float) + 1j * np.asarray(misfit["z_imag_ohm"], dtype=float)
    w = 2.0 * np.pi * freq
    z_fit = fit["rs_ohm"] + 1.0 / (1.0 / fit["rct_ohm"] + 1j * w * fit["cdl_f"])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(np.real(z_data), -np.imag(z_data), "o-", label="multiphysics")
    ax.plot(np.real(z_fit), -np.imag(z_fit), "s--", label="best 3-element fit")
    ax.set_xlabel("Z' (ohm)")
    ax.set_ylabel("-Z'' (ohm)")
    ax.set_title("Nyquist: Biofouled EIS")
    ax.grid(True, ls=":")
    ax.legend()
    fig.tight_layout()
    nyquist_path = output_dir / "multiphysics_nyquist_randles_fit.png"
    fig.savefig(nyquist_path, dpi=180)
    plt.close(fig)

    return [bode_path.name, nyquist_path.name]


def _build_markdown(summary: dict, plot_files: list[str]) -> str:
    checks = summary["checks"]
    lines = [
        "# Multiphysics Electrochem Benchmark Report",
        "",
        "## Outcome",
        "",
        f"- overall_pass: `{summary['overall_pass']}`",
        f"- cleaning_reduces_fouling: `{checks['cleaning_reduces_fouling']}`",
        f"- cleaning_reduces_film_resistance: `{checks['cleaning_reduces_film_resistance']}`",
        f"- fouling_increases_impedance: `{checks['fouling_increases_impedance']}`",
        (
            "- biofouled_not_well_fit_by_3_element_randles: "
            f"`{checks['biofouled_not_well_fit_by_3_element_randles']}`"
        ),
        "",
        "## Key Numbers",
        "",
        f"- theta_final_clean: `{summary['fouling_cleaning']['theta_final_clean']:.6f}`",
        f"- theta_final_no_clean: `{summary['fouling_cleaning']['theta_final_no_clean']:.6f}`",
        f"- rfilm_final_clean_ohm: `{summary['fouling_cleaning']['rfilm_final_clean_ohm']:.3f}`",
        f"- rfilm_final_no_clean_ohm: `{summary['fouling_cleaning']['rfilm_final_no_clean_ohm']:.3f}`",
        (
            "- randles_mean_rel_error: "
            f"`{summary['randles_misfit']['fit']['mean_rel_error']:.4f}`"
        ),
    ]
    if plot_files:
        lines.extend(
            [
                "",
                "## Plots",
                "",
                *[f"- `{name}`" for name in plot_files],
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_multiphysics_benchmarks()
    summary_path = output_dir / "multiphysics_benchmarks.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_files: list[str] = []
    if not args.no_plots:
        plot_files = _plot_impedance_comparison(
            output_dir=output_dir,
            frequencies_hz=np.asarray([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
        )

    report_path = output_dir / "multiphysics_benchmarks.md"
    report_path.write_text(_build_markdown(summary, plot_files), encoding="utf-8")

    print(json.dumps({"summary": str(summary_path), "report": str(report_path), "plots": plot_files}, indent=2))


if __name__ == "__main__":
    main()
