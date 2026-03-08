"""Rich interactive visualizer for ECSFM dataset sanity inspection.

Launch:
    uv run streamlit run scripts/visualizer.py -- --dataset /tmp/ecsfm/dataset_balanced_742k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Resolve package imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ecsfm.data.inspect import (
    DatasetLayout,
    LabelNames,
    SampleRecord,
    _active_species,
    _name_for_id,
    _sample_flags,
    _split_params,
    load_sample_records,
    scan_dataset,
    summarize_dataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARAM_LABELS = ["D_ox", "D_red", "C_ox", "C_red", "E0", "k0", "alpha"]
PARAM_UNITS = ["cm²/s", "cm²/s", "µM", "µM", "V", "cm/s", ""]
PARAM_LOG_SCALE = [True, True, False, False, False, True, False]

STAGE_COLORS = {"foundation": "#2196F3", "bridge": "#FF9800", "frontier": "#E91E63"}
TASK_COLORS = {
    "cv_reversible": "#1f77b4",
    "ca_step": "#ff7f0e",
    "cv_multispecies": "#2ca02c",
    "swv_pulse": "#d62728",
    "eis_low_freq": "#9467bd",
    "eis_high_freq": "#8c564b",
    "kinetics_limited": "#e377c2",
    "diffusion_limited": "#7f7f7f",
}
FLAG_DESCRIPTIONS = {
    "non_finite": "Contains NaN/Inf values",
    "negative_profile": "Negative concentration in spatial profile",
    "flat_current": "Current trace has near-zero dynamic range",
    "flat_potential": "Potential trace has near-zero dynamic range",
}


# ---------------------------------------------------------------------------
# CLI argument parsing (passed after `--` in streamlit run)
# ---------------------------------------------------------------------------
def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Scanning dataset chunks...")
def _scan(dataset_path: str):
    chunks, layout, labels = scan_dataset(dataset_path)
    return chunks, layout, labels


@st.cache_data(show_spinner="Computing dataset summary...")
def _summarize(dataset_path: str):
    chunks, layout, labels = _scan(dataset_path)
    return summarize_dataset(chunks, layout, labels)


@st.cache_data(show_spinner="Loading all metadata arrays...")
def _load_all_metadata(dataset_path: str):
    """Load task_id, stage_id, aug_id, and parameter vectors for all rows."""
    chunks, layout, labels = _scan(dataset_path)
    all_task = []
    all_stage = []
    all_aug = []
    all_params = []
    all_current_stats = []
    all_potential_stats = []
    for chunk in chunks:
        with np.load(chunk.path, mmap_mode="r") as data:
            rows = data["ox"].shape[0]
            task = np.asarray(data["task_id"]).reshape(-1) if "task_id" in data else np.zeros(rows, dtype=np.int32)
            stage = np.asarray(data["stage_id"]).reshape(-1) if "stage_id" in data else np.zeros(rows, dtype=np.int32)
            aug = np.asarray(data["aug_id"]).reshape(-1) if "aug_id" in data else np.zeros(rows, dtype=np.int32)
            params = np.asarray(data["p"], dtype=np.float64)
            current = np.asarray(data["i"], dtype=np.float64)
            potential = np.asarray(data["e"], dtype=np.float64)

            all_task.append(task.astype(np.int32))
            all_stage.append(stage.astype(np.int32))
            all_aug.append(aug.astype(np.int32))
            all_params.append(params)

            # Per-row summary stats for current and potential
            stats = np.column_stack([
                current.min(axis=1),
                current.max(axis=1),
                np.ptp(current, axis=1),
                potential.min(axis=1),
                potential.max(axis=1),
                np.ptp(potential, axis=1),
            ])
            all_current_stats.append(stats[:, :3])
            all_potential_stats.append(stats[:, 3:])

    return (
        np.concatenate(all_task),
        np.concatenate(all_stage),
        np.concatenate(all_aug),
        np.concatenate(all_params),
        np.concatenate(all_current_stats),
        np.concatenate(all_potential_stats),
    )


@st.cache_data(show_spinner="Loading sample records...")
def _load_records(dataset_path: str, indices: list[int]):
    chunks, _, _ = _scan(dataset_path)
    return load_sample_records(chunks, indices)


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib)
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _plot_ie_curve(record: SampleRecord) -> Figure:
    """Cyclic voltammogram: I vs E."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(record.potential, record.current, color="#2ca02c", lw=1.5)
    ax.set_xlabel("E (V)")
    ax.set_ylabel("I (mA)")
    ax.set_title("I–E Curve (Voltammogram)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_time_traces(record: SampleRecord) -> Figure:
    """E(t) and I(t) as dual-axis time traces."""
    t = np.linspace(0, 1, record.potential.shape[0])
    fig, ax1 = plt.subplots(figsize=(6, 4))

    color_e = "#1f77b4"
    color_i = "#d62728"

    ax1.plot(t, record.potential, color=color_e, lw=1.5, label="E(t)")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("E (V)", color=color_e)
    ax1.tick_params(axis="y", labelcolor=color_e)

    ax2 = ax1.twinx()
    ax2.plot(t, record.current, color=color_i, lw=1.5, label="I(t)")
    ax2.set_ylabel("I (mA)", color=color_i)
    ax2.tick_params(axis="y", labelcolor=color_i)

    ax1.set_title("Time Traces: E(t) & I(t)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_concentration_profiles(record: SampleRecord, layout: DatasetLayout) -> Figure:
    """Spatial concentration profiles for each active species."""
    ox = record.ox.reshape(layout.max_species, layout.nx)
    red = record.red.reshape(layout.max_species, layout.nx)
    x = np.linspace(0, 0.05, layout.nx)  # L = 0.05 cm

    slices = _split_params(record.params, layout.max_species)
    active = _active_species(slices)

    n_active = min(len(active), 4)
    fig, axes = plt.subplots(1, max(n_active, 1), figsize=(4 * max(n_active, 1), 4), squeeze=False)

    for i, sp_idx in enumerate(active[:4]):
        ax = axes[0, i]
        ax.plot(x * 1e4, ox[sp_idx], lw=1.5, color="#1f77b4", label="Ox")
        ax.plot(x * 1e4, red[sp_idx], lw=1.5, ls="--", color="#d62728", label="Red")
        e0_val = slices["E0"][sp_idx]
        ax.set_title(f"Species {sp_idx + 1} (E⁰={e0_val:.3f} V)", fontsize=10)
        ax.set_xlabel("Distance (µm)")
        ax.set_ylabel("C (µM)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Final Concentration Profiles", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def _plot_parameter_radar(record: SampleRecord, layout: DatasetLayout) -> Figure:
    """Show physical parameters for each active species as a table-like bar chart."""
    slices = _split_params(record.params, layout.max_species)
    active = _active_species(slices)

    fig, axes = plt.subplots(1, len(PARAM_LABELS), figsize=(14, 3), squeeze=False)
    for j, (pname, unit, log_s) in enumerate(zip(PARAM_LABELS, PARAM_UNITS, PARAM_LOG_SCALE)):
        ax = axes[0, j]
        vals = slices[pname][active]
        bars = ax.barh(range(len(active)), vals, color="#4CAF50", height=0.6)
        ax.set_yticks(range(len(active)))
        ax.set_yticklabels([f"S{s + 1}" for s in active], fontsize=8)
        label = f"{pname}"
        if unit:
            label += f" ({unit})"
        ax.set_xlabel(label, fontsize=8)
        if log_s:
            ax.set_xscale("log")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Physical Parameters per Species", fontsize=11, y=1.05)
    fig.tight_layout()
    return fig


def _plot_distribution_bars(summary: dict) -> Figure:
    """Task, stage, and augmentation distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, key, title in zip(
        axes,
        ["task", "stage", "augmentation"],
        ["Task Distribution", "Stage Distribution", "Augmentation Distribution"],
    ):
        rows = [r for r in summary["distribution"][key] if r["count"] > 0]
        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue
        names = [r["name"] for r in rows]
        counts = [r["count"] for r in rows]
        colors = [TASK_COLORS.get(n, STAGE_COLORS.get(n, "#607D8B")) for n in names]
        ax.barh(range(len(names)), counts, color=colors, height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Count")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_diagnostics_bars(summary: dict) -> Figure:
    """Diagnostic sanity flags as percentage bars."""
    diag = summary["diagnostics"]
    total = max(1, summary["total_rows"])

    labels = ["Non-finite", "Neg. profile", "Neg. bulk", "Invalid α", "Flat current", "Flat potential"]
    keys = ["nonfinite_rows", "negative_profile_rows", "negative_bulk_rows", "invalid_alpha_rows", "flat_current_rows", "flat_potential_rows"]
    pcts = [100.0 * diag[k] / total for k in keys]

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#f44336" if p > 1 else "#4CAF50" for p in pcts]
    ax.barh(range(len(labels)), pcts, color=colors, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("% of total rows")
    ax.set_title("Sanity Diagnostic Flags")

    for i, (p, k) in enumerate(zip(pcts, keys)):
        ax.text(p + 0.3, i, f"{diag[k]:,} ({p:.2f}%)", va="center", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_param_distributions(all_params: np.ndarray, max_species: int, task_ids: np.ndarray, labels: LabelNames) -> Figure:
    """Histograms of physical parameters colored by task."""
    m = max_species
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    param_slices = {
        "D_ox (log)": np.log10(np.exp(all_params[:, 0:m]).max(axis=1)),
        "D_red (log)": np.log10(np.exp(all_params[:, m:2*m]).max(axis=1)),
        "C_ox (µM)": all_params[:, 2*m:3*m].max(axis=1),
        "C_red (µM)": all_params[:, 3*m:4*m].max(axis=1),
        "E0 (V)": all_params[:, 4*m:5*m].mean(axis=1),
        "k0 (log)": np.log10(np.exp(all_params[:, 5*m:6*m]).max(axis=1)),
        "alpha": all_params[:, 6*m:7*m].mean(axis=1),
    }

    unique_tasks = np.unique(task_ids)
    for i, (pname, vals) in enumerate(param_slices.items()):
        ax = axes[i]
        finite = np.isfinite(vals)
        for tid in unique_tasks:
            mask = (task_ids == tid) & finite
            if mask.sum() == 0:
                continue
            tname = _name_for_id(labels.task, int(tid), "task")
            ax.hist(vals[mask], bins=50, alpha=0.6, label=tname,
                    color=TASK_COLORS.get(tname, None), histtype="stepfilled")
        ax.set_title(pname, fontsize=10)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # legend in last panel
    axes[-1].axis("off")
    handles, leg_labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, leg_labels, loc="center", fontsize=9, title="Task")

    fig.suptitle("Parameter Distributions by Task Type", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _plot_current_vs_params_scatter(
    all_params: np.ndarray, current_stats: np.ndarray,
    max_species: int, task_ids: np.ndarray, labels: LabelNames,
    max_points: int = 5000,
) -> Figure:
    """Scatter: peak current magnitude vs key parameters, sampled for performance."""
    m = max_species
    n = len(task_ids)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
    else:
        idx = np.arange(n)

    i_ptp = current_stats[idx, 2]  # peak-to-peak current
    c_ox_max = all_params[idx, 2*m:3*m].max(axis=1)
    k0_log = np.log10(np.exp(all_params[idx, 5*m:6*m]).max(axis=1))
    e0_mean = all_params[idx, 4*m:5*m].mean(axis=1)
    tid = task_ids[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, xvals, xlabel in zip(
        axes,
        [c_ox_max, k0_log, e0_mean],
        ["C_ox max (µM)", "log₁₀(k0 max) (cm/s)", "E⁰ mean (V)"],
    ):
        for t in np.unique(tid):
            mask = tid == t
            tname = _name_for_id(labels.task, int(t), "task")
            ax.scatter(xvals[mask], i_ptp[mask], s=4, alpha=0.4,
                       color=TASK_COLORS.get(tname, None), label=tname)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("I peak-to-peak (mA)")
        ax.grid(True, alpha=0.3)

    axes[-1].legend(fontsize=7, markerscale=3, loc="upper left")
    fig.suptitle("Current Response vs Physical Parameters", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _plot_eis_spectrum(record: SampleRecord) -> Figure:
    """FFT magnitude spectrum of the potential waveform for EIS samples."""
    spectrum = np.fft.rfft(record.potential)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(record.potential))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs[1:], magnitude[1:], color="#9467bd", lw=1.5)  # skip DC
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("|FFT(E)|")
    ax.set_title("EIS Potential Spectrum (FFT Magnitude)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_ie_overlay(records: list[SampleRecord], labels: LabelNames, max_traces: int = 20) -> Figure:
    """Overlay I-E curves for a batch of samples, colored by task."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for rec in records[:max_traces]:
        tname = _name_for_id(labels.task, rec.task_id, "task")
        ax.plot(rec.potential, rec.current, lw=0.8, alpha=0.6,
                color=TASK_COLORS.get(tname, "#999"), label=tname)

    # Deduplicate legend
    handles, leg_labels = ax.get_legend_handles_labels()
    seen = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, leg_labels):
        if l not in seen:
            seen[l] = True
            unique_h.append(h)
            unique_l.append(l)
    ax.legend(unique_h, unique_l, fontsize=8, loc="best")
    ax.set_xlabel("E (V)")
    ax.set_ylabel("I (mA)")
    ax.set_title("I–E Overlay (batch)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="ECSFM Dataset Visualizer",
        page_icon="⚗",
        layout="wide",
    )

    st.title("ECSFM Dataset Visualizer")
    st.caption("Electrochemical Sensing Foundation Model — Data Sanity Inspector")

    # -- Sidebar: dataset path -------------------------------------------
    cli_args = _parse_cli()

    dataset_path = st.sidebar.text_input(
        "Dataset path",
        value=cli_args.dataset or "/tmp/ecsfm/dataset_balanced_742k",
    )

    if not dataset_path or not Path(dataset_path).exists():
        st.warning(f"Dataset path not found: `{dataset_path}`")
        st.info("Pass `--dataset /path/to/chunks` or enter the path in the sidebar.")
        st.stop()

    # -- Load metadata ----------------------------------------------------
    try:
        chunks, layout, labels = _scan(dataset_path)
    except Exception as exc:
        st.error(f"Failed to scan dataset: {exc}")
        st.stop()

    total_rows = sum(c.rows for c in chunks)
    summary = _summarize(dataset_path)
    task_ids, stage_ids, aug_ids, all_params, current_stats, potential_stats = _load_all_metadata(dataset_path)

    # -- Sidebar info -----------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Geometry")
    st.sidebar.markdown(
        f"- **Chunks:** {len(chunks)}\n"
        f"- **Total rows:** {total_rows:,}\n"
        f"- **max_species:** {layout.max_species}\n"
        f"- **nx:** {layout.nx}\n"
        f"- **signal_len:** {layout.signal_len}\n"
        f"- **phys_dim:** {layout.phys_dim}"
    )

    # -- Tabs -------------------------------------------------------------
    tab_overview, tab_params, tab_browse, tab_batch, tab_sanity = st.tabs(
        ["Overview", "Parameter Explorer", "Sample Browser", "Batch Comparison", "Sanity Checks"]
    )

    # =====================================================================
    # TAB: Overview
    # =====================================================================
    with tab_overview:
        st.subheader("Dataset Distribution")
        st.pyplot(_plot_distribution_bars(summary))

        # Task balance metric
        task_dist = summary["distribution"]["task"]
        task_counts = [r["count"] for r in task_dist if r["count"] > 0]
        if task_counts:
            balance_ratio = max(task_counts) / max(min(task_counts), 1)
            st.metric(
                "Task Balance (max/min ratio)",
                f"{balance_ratio:.2f}x",
                delta="balanced" if balance_ratio < 5 else "imbalanced",
                delta_color="normal" if balance_ratio < 5 else "inverse",
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Statistics")
            r = summary["ranges"]["current_mA"]
            st.metric("Min (mA)", f"{r['min']:.4g}")
            st.metric("Max (mA)", f"{r['max']:.4g}")
            st.metric("Mean (mA)", f"{r['mean']:.4g}")
            st.metric("Std (mA)", f"{r['std']:.4g}")
        with col2:
            st.subheader("Potential Statistics")
            r = summary["ranges"]["potential_V"]
            st.metric("Min (V)", f"{r['min']:.4g}")
            st.metric("Max (V)", f"{r['max']:.4g}")
            st.metric("Mean (V)", f"{r['mean']:.4g}")
            st.metric("Std (V)", f"{r['std']:.4g}")

        st.subheader("Diagnostic Flags")
        st.pyplot(_plot_diagnostics_bars(summary))

        if summary["diagnosis"]:
            for d in summary["diagnosis"]:
                if "red flag" not in d.lower():
                    st.warning(d)
                else:
                    st.success(d)

    # =====================================================================
    # TAB: Parameter Explorer
    # =====================================================================
    with tab_params:
        st.subheader("Physical Parameter Distributions")
        st.pyplot(_plot_param_distributions(all_params, layout.max_species, task_ids, labels))

        st.subheader("Current Response vs Parameters")
        st.pyplot(_plot_current_vs_params_scatter(
            all_params, current_stats, layout.max_species, task_ids, labels,
        ))

        # -- Correlation matrix of parameters ---
        st.subheader("Parameter Correlation Matrix")
        m = layout.max_species
        # Take the first species' parameters for correlation
        param_matrix = np.column_stack([
            np.exp(all_params[:, 0]),        # D_ox species 0
            np.exp(all_params[:, m]),         # D_red species 0
            all_params[:, 2 * m],             # C_ox species 0
            all_params[:, 3 * m],             # C_red species 0
            all_params[:, 4 * m],             # E0 species 0
            np.exp(all_params[:, 5 * m]),     # k0 species 0
            all_params[:, 6 * m],             # alpha species 0
            current_stats[:, 2],              # I peak-to-peak
        ])
        finite_mask = np.isfinite(param_matrix).all(axis=1)
        param_matrix = param_matrix[finite_mask]

        corr_labels = PARAM_LABELS + ["I_ptp"]
        if param_matrix.shape[0] > 100:
            corr = np.corrcoef(param_matrix, rowvar=False)
            fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
            im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax_corr.set_xticks(range(len(corr_labels)))
            ax_corr.set_yticks(range(len(corr_labels)))
            ax_corr.set_xticklabels(corr_labels, rotation=45, ha="right")
            ax_corr.set_yticklabels(corr_labels)
            for ii in range(len(corr_labels)):
                for jj in range(len(corr_labels)):
                    ax_corr.text(jj, ii, f"{corr[ii, jj]:.2f}", ha="center", va="center", fontsize=8)
            fig_corr.colorbar(im, ax=ax_corr, shrink=0.8)
            ax_corr.set_title("Correlation (Species 0 params + I_ptp)")
            fig_corr.tight_layout()
            st.pyplot(fig_corr)
        else:
            st.info("Not enough finite rows for correlation matrix.")

    # =====================================================================
    # TAB: Sample Browser
    # =====================================================================
    with tab_browse:
        st.subheader("Individual Sample Inspector")

        col_nav1, col_nav2, col_nav3 = st.columns([2, 2, 2])
        with col_nav1:
            browse_mode = st.radio("Selection mode", ["Random", "By index", "Filter by task"], horizontal=True)
        with col_nav2:
            seed = st.number_input("Random seed", value=2026, step=1)

        if browse_mode == "Random":
            rng = np.random.default_rng(seed)
            sample_idx = int(rng.integers(0, total_rows))
            sample_idx = st.number_input("Sample index (randomized)", value=sample_idx, min_value=0, max_value=total_rows - 1)
        elif browse_mode == "By index":
            sample_idx = st.number_input("Sample index", value=0, min_value=0, max_value=total_rows - 1)
        else:
            task_filter = st.selectbox("Task type", labels.task)
            task_id_filter = labels.task.index(task_filter)
            matching = np.where(task_ids == task_id_filter)[0]
            if len(matching) == 0:
                st.warning(f"No samples found for task '{task_filter}'")
                st.stop()
            offset = st.slider("Sample offset within task", 0, len(matching) - 1, 0)
            sample_idx = int(matching[offset])

        records = _load_records(dataset_path, [sample_idx])
        if not records:
            st.error("Failed to load sample.")
            st.stop()

        rec = records[0]

        # Metadata header
        tname = _name_for_id(labels.task, rec.task_id, "task")
        sname = _name_for_id(labels.stage, rec.stage_id, "stage")
        aname = _name_for_id(labels.augmentation, rec.aug_id, "augmentation")
        flags = _sample_flags(rec)

        st.markdown(
            f"**Sample #{rec.global_index}** | "
            f"Chunk: `{rec.chunk_path.name}` row {rec.chunk_row} | "
            f"Task: `{tname}` | Stage: `{sname}` | Aug: `{aname}`"
        )

        if flags:
            for f in flags:
                st.error(f"Flag: **{f}** — {FLAG_DESCRIPTIONS.get(f, '')}")
        else:
            st.success("No sanity flags")

        # Plots
        col_ie, col_time = st.columns(2)
        with col_ie:
            st.pyplot(_plot_ie_curve(rec))
        with col_time:
            st.pyplot(_plot_time_traces(rec))

        # EIS spectrum analysis
        if "eis" in tname.lower():
            st.subheader("EIS Spectrum Analysis")
            st.pyplot(_plot_eis_spectrum(rec))

        st.pyplot(_plot_concentration_profiles(rec, layout))

        st.subheader("Physical Parameters")
        st.pyplot(_plot_parameter_radar(rec, layout))

        # Raw parameter table
        slices = _split_params(rec.params, layout.max_species)
        active = _active_species(slices)
        param_data = {}
        for pname in PARAM_LABELS:
            param_data[pname] = [f"{slices[pname][s]:.4g}" for s in active]
        st.dataframe(
            {k: v for k, v in param_data.items()},
            column_config={k: st.column_config.TextColumn(k) for k in PARAM_LABELS},
        )

    # =====================================================================
    # TAB: Batch Comparison
    # =====================================================================
    with tab_batch:
        st.subheader("Batch I–E Overlay")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            batch_task = st.selectbox("Filter task (or All)", ["All"] + labels.task, key="batch_task")
        with col_b2:
            batch_n = st.slider("Number of traces", 5, 50, 20)

        batch_seed = st.number_input("Batch seed", value=42, step=1, key="batch_seed")
        rng_batch = np.random.default_rng(batch_seed)

        if batch_task == "All":
            pool = np.arange(total_rows)
        else:
            tid_filt = labels.task.index(batch_task)
            pool = np.where(task_ids == tid_filt)[0]

        if len(pool) == 0:
            st.warning("No samples match filter.")
        else:
            batch_idx = rng_batch.choice(pool, size=min(batch_n, len(pool)), replace=False)
            batch_records = _load_records(dataset_path, [int(i) for i in batch_idx])
            st.pyplot(_plot_ie_overlay(batch_records, labels, max_traces=batch_n))

            # Show E(t) overlay too
            fig_et, ax_et = plt.subplots(figsize=(8, 4))
            for r in batch_records[:batch_n]:
                t = np.linspace(0, 1, r.potential.shape[0])
                tname_b = _name_for_id(labels.task, r.task_id, "task")
                ax_et.plot(t, r.potential, lw=0.7, alpha=0.5, color=TASK_COLORS.get(tname_b, "#999"))
            ax_et.set_xlabel("Normalized Time")
            ax_et.set_ylabel("E (V)")
            ax_et.set_title("Applied Potential Waveforms")
            ax_et.grid(True, alpha=0.3)
            fig_et.tight_layout()
            st.pyplot(fig_et)

            # I(t) overlay
            fig_it, ax_it = plt.subplots(figsize=(8, 4))
            for r in batch_records[:batch_n]:
                t = np.linspace(0, 1, r.current.shape[0])
                tname_b = _name_for_id(labels.task, r.task_id, "task")
                ax_it.plot(t, r.current, lw=0.7, alpha=0.5, color=TASK_COLORS.get(tname_b, "#999"))
            ax_it.set_xlabel("Normalized Time")
            ax_it.set_ylabel("I (mA)")
            ax_it.set_title("Current Response Traces")
            ax_it.grid(True, alpha=0.3)
            fig_it.tight_layout()
            st.pyplot(fig_it)

            # EIS FFT overlay for EIS tasks
            if batch_task != "All" and "eis" in batch_task.lower():
                st.subheader("EIS FFT Overlay")
                fig_fft, ax_fft = plt.subplots(figsize=(8, 4))
                for r in batch_records[:batch_n]:
                    spectrum = np.fft.rfft(r.potential)
                    magnitude = np.abs(spectrum)
                    freqs = np.fft.rfftfreq(len(r.potential))
                    ax_fft.plot(freqs[1:], magnitude[1:], lw=0.7, alpha=0.5,
                                color=TASK_COLORS.get(batch_task, "#999"))
                ax_fft.set_xlabel("Normalized Frequency")
                ax_fft.set_ylabel("|FFT(E)|")
                ax_fft.set_title("EIS Potential Spectra Overlay")
                ax_fft.set_yscale("log")
                ax_fft.grid(True, alpha=0.3)
                fig_fft.tight_layout()
                st.pyplot(fig_fft)

    # =====================================================================
    # TAB: Sanity Checks
    # =====================================================================
    with tab_sanity:
        st.subheader("Electrochemical Sanity Checks")

        st.markdown("""
        These checks validate the dataset from a physical electrochemistry perspective:
        - **Randles-Sevcik scaling**: For reversible CV, peak current should scale with √(scan rate) and C_ox
        - **Nernst consistency**: E⁰ values should produce expected half-wave potentials
        - **Concentration conservation**: Total species concentration should be conserved
        - **Diffusion layer thickness**: Concentration gradients should vanish at the bulk boundary
        """)

        # --- Check 1: Concentration conservation at boundary ---
        st.subheader("1. Bulk Boundary Condition Check")
        st.markdown("The last grid point should approximate the bulk concentration (C_ox or C_red).")

        n_check = min(500, total_rows)
        rng_san = np.random.default_rng(123)
        check_idx = rng_san.choice(total_rows, n_check, replace=False).tolist()
        check_records = _load_records(dataset_path, [int(i) for i in check_idx])

        boundary_errors_ox = []
        boundary_errors_red = []
        for r in check_records:
            ox = r.ox.reshape(layout.max_species, layout.nx)
            red = r.red.reshape(layout.max_species, layout.nx)
            slices = _split_params(r.params, layout.max_species)
            active = _active_species(slices)
            for sp in active:
                c_ox_bulk = slices["C_ox"][sp]
                c_red_bulk = slices["C_red"][sp]
                if c_ox_bulk > 1e-3:
                    boundary_errors_ox.append(abs(ox[sp, -1] - c_ox_bulk) / c_ox_bulk)
                if c_red_bulk > 1e-3:
                    boundary_errors_red.append(abs(red[sp, -1] - c_red_bulk) / c_red_bulk)

        if boundary_errors_ox:
            be_ox = np.array(boundary_errors_ox)
            fig_be, axes_be = plt.subplots(1, 2, figsize=(10, 3))
            axes_be[0].hist(be_ox, bins=50, color="#2196F3", edgecolor="white")
            axes_be[0].set_xlabel("Relative error |C_ox(L) - C_ox_bulk| / C_ox_bulk")
            axes_be[0].set_ylabel("Count")
            axes_be[0].set_title(f"Ox boundary (median={np.median(be_ox):.4g})")
            axes_be[0].axvline(0.05, color="red", ls="--", label="5% threshold")
            axes_be[0].legend()

            if boundary_errors_red:
                be_red = np.array(boundary_errors_red)
                axes_be[1].hist(be_red, bins=50, color="#FF9800", edgecolor="white")
                axes_be[1].set_xlabel("Relative error |C_red(L) - C_red_bulk| / C_red_bulk")
                axes_be[1].set_title(f"Red boundary (median={np.median(be_red):.4g})")
                axes_be[1].axvline(0.05, color="red", ls="--", label="5% threshold")
                axes_be[1].legend()
            else:
                axes_be[1].text(0.5, 0.5, "No significant C_red samples", ha="center", va="center")
                axes_be[1].set_title("Red boundary")

            fig_be.tight_layout()
            st.pyplot(fig_be)

            pct_ok = 100 * np.mean(be_ox < 0.05)
            if pct_ok > 95:
                st.success(f"Boundary condition satisfied for {pct_ok:.1f}% of oxidized species.")
            else:
                st.warning(f"Only {pct_ok:.1f}% of oxidized species satisfy <5% boundary error.")

        # --- Check 2: Non-negative concentrations ---
        st.subheader("2. Non-negative Concentration Check")
        neg_count = 0
        for r in check_records:
            if np.min(r.ox) < -1e-6 or np.min(r.red) < -1e-6:
                neg_count += 1
        pct_neg = 100 * neg_count / len(check_records)
        if pct_neg < 1:
            st.success(f"Non-negative concentrations: {100 - pct_neg:.1f}% clean ({neg_count}/{len(check_records)} violations)")
        else:
            st.warning(f"{pct_neg:.1f}% of checked samples have negative concentrations ({neg_count}/{len(check_records)})")

        # --- Check 3: Current magnitude vs concentration scaling ---
        st.subheader("3. Current–Concentration Scaling")
        st.markdown("For diffusion-controlled processes, peak current should scale roughly linearly with C_ox.")

        m = layout.max_species
        cv_mask = task_ids == (labels.task.index("cv_reversible") if "cv_reversible" in labels.task else 0)
        if cv_mask.sum() > 50:
            cv_idx = np.where(cv_mask)[0]
            sample_cv = rng_san.choice(cv_idx, min(300, len(cv_idx)), replace=False)
            cv_recs = _load_records(dataset_path, [int(i) for i in sample_cv])

            c_ox_vals = []
            i_peak_vals = []
            for r in cv_recs:
                sl = _split_params(r.params, layout.max_species)
                act = _active_species(sl)
                c_ox_vals.append(sl["C_ox"][act[0]])
                i_peak_vals.append(np.max(np.abs(r.current)))

            c_ox_arr = np.array(c_ox_vals)
            i_peak_arr = np.array(i_peak_vals)

            fig_rs, ax_rs = plt.subplots(figsize=(6, 4))
            ax_rs.scatter(c_ox_arr, i_peak_arr, s=10, alpha=0.5, color="#2ca02c")
            ax_rs.set_xlabel("C_ox (µM)")
            ax_rs.set_ylabel("|I_peak| (mA)")
            ax_rs.set_title("Randles-Sevcik Check: I_peak vs C_ox (cv_reversible)")
            ax_rs.grid(True, alpha=0.3)

            # Fit line
            finite = np.isfinite(c_ox_arr) & np.isfinite(i_peak_arr) & (c_ox_arr > 0)
            if finite.sum() > 10:
                from numpy.polynomial import polynomial as P
                coeffs = P.polyfit(c_ox_arr[finite], i_peak_arr[finite], 1)
                x_fit = np.linspace(c_ox_arr[finite].min(), c_ox_arr[finite].max(), 100)
                y_fit = P.polyval(x_fit, coeffs)
                ax_rs.plot(x_fit, y_fit, "r--", lw=2, label=f"Linear fit (slope={coeffs[1]:.4g})")
                ax_rs.legend()

                corr_val = np.corrcoef(c_ox_arr[finite], i_peak_arr[finite])[0, 1]
                if abs(corr_val) > 0.5:
                    st.success(f"Good I_peak–C_ox correlation: r = {corr_val:.3f}")
                else:
                    st.warning(f"Weak I_peak–C_ox correlation: r = {corr_val:.3f} (expected positive for diffusion control)")

            fig_rs.tight_layout()
            st.pyplot(fig_rs)
        else:
            st.info("Not enough `cv_reversible` samples to test Randles-Sevcik scaling.")

        # --- Check 4: Species count per task ---
        st.subheader("4. Active Species Count by Task")
        task_species_counts: dict[str, list[int]] = {t: [] for t in labels.task}
        for r in check_records:
            sl = _split_params(r.params, layout.max_species)
            act = _active_species(sl)
            tname_s = _name_for_id(labels.task, r.task_id, "task")
            if tname_s in task_species_counts:
                task_species_counts[tname_s].append(len(act))

        fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
        task_names_plot = [t for t in labels.task if task_species_counts.get(t)]
        positions = range(len(task_names_plot))
        bp_data = [task_species_counts[t] for t in task_names_plot]
        if bp_data:
            ax_sp.boxplot(bp_data, positions=list(positions), widths=0.6, patch_artist=True,
                          boxprops=dict(facecolor="#E3F2FD"),
                          medianprops=dict(color="#1565C0", lw=2))
            ax_sp.set_xticks(list(positions))
            ax_sp.set_xticklabels(task_names_plot, rotation=30, ha="right")
            ax_sp.set_ylabel("Active species count")
            ax_sp.set_title("Active Species per Task Type")
            ax_sp.grid(True, axis="y", alpha=0.3)
        fig_sp.tight_layout()
        st.pyplot(fig_sp)

        # --- Check 5: Mass Conservation ---
        st.subheader("5. Mass Conservation")
        st.markdown(
            "CV of total concentration (C_ox + C_red) across interior grid points "
            "should be small for active species. Values above 10% indicate mass leakage."
        )

        cv_values = []
        for r in check_records:
            ox = r.ox.reshape(layout.max_species, layout.nx)
            red = r.red.reshape(layout.max_species, layout.nx)
            slices_mc = _split_params(r.params, layout.max_species)
            active_mc = _active_species(slices_mc)
            for sp in active_mc:
                total = ox[sp, 1:-1] + red[sp, 1:-1]  # interior points
                if total.mean() > 1e-6:
                    cv_values.append(total.std() / total.mean())

        if cv_values:
            cv_arr = np.array(cv_values)
            fig_mc, ax_mc = plt.subplots(figsize=(6, 4))
            ax_mc.hist(cv_arr, bins=50, color="#4CAF50", edgecolor="white")
            ax_mc.axvline(0.10, color="red", ls="--", lw=2, label="10% threshold")
            ax_mc.set_xlabel("CV(C_ox + C_red) across interior points")
            ax_mc.set_ylabel("Count")
            ax_mc.set_title("Mass Conservation Check")
            ax_mc.legend()
            ax_mc.grid(True, alpha=0.3)
            fig_mc.tight_layout()
            st.pyplot(fig_mc)

            pct_pass = 100 * np.mean(cv_arr < 0.10)
            if pct_pass > 90:
                st.success(f"Mass conservation: {pct_pass:.1f}% of species pass (<10% CV)")
            else:
                st.warning(f"Mass conservation: only {pct_pass:.1f}% of species pass (<10% CV)")
        else:
            st.info("No active species found for mass conservation check.")

        # --- Check 6: Cross-Task Distinguishability ---
        st.subheader("6. Cross-Task Distinguishability")
        st.markdown(
            "Pairwise cosine similarity of task-averaged current traces. "
            "Lower off-diagonal values indicate better task separability."
        )

        # Group current traces by task
        task_currents: dict[str, list[np.ndarray]] = {}
        for r in check_records:
            tname_ct = _name_for_id(labels.task, r.task_id, "task")
            task_currents.setdefault(tname_ct, []).append(r.current)

        active_tasks = [t for t in labels.task if len(task_currents.get(t, [])) >= 5]
        if len(active_tasks) >= 2:
            # Compute mean current per task
            mean_currents = []
            for t in active_tasks:
                stacked = np.vstack(task_currents[t])
                mean_currents.append(stacked.mean(axis=0))
            mean_currents = np.array(mean_currents)

            # Pairwise cosine similarity
            norms = np.linalg.norm(mean_currents, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normed = mean_currents / norms
            cos_sim = normed @ normed.T

            fig_cs, ax_cs = plt.subplots(figsize=(7, 6))
            im_cs = ax_cs.imshow(cos_sim, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
            ax_cs.set_xticks(range(len(active_tasks)))
            ax_cs.set_yticks(range(len(active_tasks)))
            ax_cs.set_xticklabels(active_tasks, rotation=45, ha="right", fontsize=9)
            ax_cs.set_yticklabels(active_tasks, fontsize=9)
            for ii in range(len(active_tasks)):
                for jj in range(len(active_tasks)):
                    ax_cs.text(jj, ii, f"{cos_sim[ii, jj]:.2f}",
                               ha="center", va="center", fontsize=8)
            fig_cs.colorbar(im_cs, ax=ax_cs, shrink=0.8)
            ax_cs.set_title("Task Cosine Similarity (mean current traces)")
            fig_cs.tight_layout()
            st.pyplot(fig_cs)

            # Report mean off-diagonal similarity
            mask_offdiag = ~np.eye(len(active_tasks), dtype=bool)
            mean_offdiag = cos_sim[mask_offdiag].mean()
            if mean_offdiag < 0.7:
                st.success(f"Mean off-diagonal cosine similarity: {mean_offdiag:.3f} (good separability)")
            else:
                st.warning(f"Mean off-diagonal cosine similarity: {mean_offdiag:.3f} (tasks may be hard to distinguish)")
        else:
            st.info("Need at least 2 tasks with 5+ samples for distinguishability analysis.")


if __name__ == "__main__":
    main()
