from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

REQUIRED_KEYS = ("ox", "red", "i", "e", "p")

DEFAULT_TASK_NAMES = [
    "cv_reversible",
    "ca_step",
    "cv_multispecies",
    "swv_pulse",
    "eis_low_freq",
    "eis_high_freq",
    "kinetics_limited",
    "diffusion_limited",
]
DEFAULT_STAGE_NAMES = ["foundation", "bridge", "frontier"]
DEFAULT_AUGMENTATION_NAMES = ["none", "permute_species", "scale_concentration"]

CHUNK_RE = re.compile(r"^chunk_(\d+)\.npz$")


@dataclass(frozen=True)
class ChunkSpec:
    path: Path
    rows: int


@dataclass(frozen=True)
class DatasetLayout:
    max_species: int
    nx: int
    signal_len: int
    state_width: int
    phys_dim: int


@dataclass(frozen=True)
class LabelNames:
    task: list[str]
    stage: list[str]
    augmentation: list[str]


@dataclass(frozen=True)
class SampleRecord:
    global_index: int
    chunk_path: Path
    chunk_row: int
    ox: np.ndarray
    red: np.ndarray
    current: np.ndarray
    potential: np.ndarray
    params: np.ndarray
    task_id: int
    stage_id: int
    aug_id: int


def _decode_names(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in np.asarray(values).tolist():
        if isinstance(value, (bytes, np.bytes_)):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _name_for_id(names: list[str], idx: int, prefix: str) -> str:
    if 0 <= idx < len(names):
        return names[idx]
    return f"{prefix}_{idx}"


def _load_optional_ids(data: np.lib.npyio.NpzFile, key: str, rows: int) -> np.ndarray:
    if key in data:
        values = np.asarray(data[key]).reshape(-1)
        if values.shape[0] != rows:
            raise ValueError(f"Key '{key}' length mismatch: {values.shape[0]} != {rows}")
        return values.astype(np.int32)
    return np.zeros((rows,), dtype=np.int32)


def _chunk_sort_key(path: Path) -> tuple[int, int | str]:
    match = CHUNK_RE.match(path.name)
    if match:
        return 0, int(match.group(1))
    return 1, path.name


def resolve_chunk_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Dataset path must be a file or directory: {path}")

    chunk_files = sorted(path.glob("chunk_*.npz"), key=_chunk_sort_key)
    if not chunk_files:
        chunk_files = sorted(path.glob("*.npz"), key=_chunk_sort_key)
    if not chunk_files:
        raise FileNotFoundError(f"No .npz files found in {path}")
    return chunk_files


def _infer_layout_from_arrays(
    ox: np.ndarray,
    red: np.ndarray,
    current: np.ndarray,
    potential: np.ndarray,
    params: np.ndarray,
) -> DatasetLayout:
    if ox.ndim != 2 or red.ndim != 2 or current.ndim != 2 or potential.ndim != 2 or params.ndim != 2:
        raise ValueError("Expected dataset arrays to be 2D")
    if ox.shape != red.shape:
        raise ValueError(f"ox and red shape mismatch: {ox.shape} vs {red.shape}")
    if current.shape != potential.shape:
        raise ValueError(f"i and e shape mismatch: {current.shape} vs {potential.shape}")
    if params.shape[1] % 7 != 0:
        raise ValueError(f"Physical parameter width must be divisible by 7, got {params.shape[1]}")

    max_species = params.shape[1] // 7
    if max_species <= 0:
        raise ValueError("Inferred max_species <= 0")
    if ox.shape[1] % max_species != 0:
        raise ValueError(
            f"State width {ox.shape[1]} is not divisible by max_species {max_species}"
        )

    nx = ox.shape[1] // max_species
    return DatasetLayout(
        max_species=max_species,
        nx=nx,
        signal_len=current.shape[1],
        state_width=ox.shape[1],
        phys_dim=params.shape[1],
    )


def _check_layout(expected: DatasetLayout, actual: DatasetLayout, chunk_path: Path) -> None:
    if expected != actual:
        raise ValueError(
            f"Inconsistent layout in {chunk_path}: expected {expected}, got {actual}"
        )


def scan_dataset(dataset_path: str | Path) -> tuple[list[ChunkSpec], DatasetLayout, LabelNames]:
    chunk_paths = resolve_chunk_files(dataset_path)
    if not chunk_paths:
        raise ValueError("No chunks resolved")

    chunk_specs: list[ChunkSpec] = []
    inferred_layout: DatasetLayout | None = None
    task_names: list[str] | None = None
    stage_names: list[str] | None = None
    augmentation_names: list[str] | None = None

    for chunk_path in chunk_paths:
        with np.load(chunk_path, mmap_mode="r") as data:
            for key in REQUIRED_KEYS:
                if key not in data:
                    raise KeyError(f"Chunk {chunk_path} missing required key '{key}'")

            ox = np.asarray(data["ox"])
            red = np.asarray(data["red"])
            current = np.asarray(data["i"])
            potential = np.asarray(data["e"])
            params = np.asarray(data["p"])
            rows = int(ox.shape[0])
            if not (rows == red.shape[0] == current.shape[0] == potential.shape[0] == params.shape[0]):
                raise ValueError(f"Chunk {chunk_path} has inconsistent row counts")

            layout = _infer_layout_from_arrays(ox, red, current, potential, params)
            if inferred_layout is None:
                inferred_layout = layout
            else:
                _check_layout(inferred_layout, layout, chunk_path)

            if task_names is None and "task_names" in data:
                task_names = _decode_names(np.asarray(data["task_names"]))
            if stage_names is None and "stage_names" in data:
                stage_names = _decode_names(np.asarray(data["stage_names"]))
            if augmentation_names is None and "augmentation_names" in data:
                augmentation_names = _decode_names(np.asarray(data["augmentation_names"]))

            chunk_specs.append(ChunkSpec(path=chunk_path, rows=rows))

    if inferred_layout is None:
        raise ValueError("Failed to infer dataset layout")

    labels = LabelNames(
        task=task_names if task_names is not None else DEFAULT_TASK_NAMES.copy(),
        stage=stage_names if stage_names is not None else DEFAULT_STAGE_NAMES.copy(),
        augmentation=(
            augmentation_names
            if augmentation_names is not None
            else DEFAULT_AUGMENTATION_NAMES.copy()
        ),
    )
    return chunk_specs, inferred_layout, labels


def select_random_global_indices(total_rows: int, n_samples: int, seed: int) -> list[int]:
    if total_rows <= 0:
        raise ValueError(f"total_rows must be > 0, got {total_rows}")
    if n_samples <= 0:
        return []
    rng = np.random.default_rng(seed)
    replace = total_rows < n_samples
    draws = rng.choice(total_rows, size=n_samples, replace=replace)
    return [int(v) for v in draws.tolist()]


def parse_global_indices(raw: str, total_rows: int) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("sample-indices was provided but empty")

    out: list[int] = []
    for token in values:
        idx = int(token)
        if idx < 0 or idx >= total_rows:
            raise ValueError(f"sample index {idx} out of range [0, {total_rows - 1}]")
        out.append(idx)
    return out


def _build_chunk_offsets(chunks: list[ChunkSpec]) -> np.ndarray:
    if not chunks:
        raise ValueError("chunks cannot be empty")
    return np.cumsum(np.asarray([chunk.rows for chunk in chunks], dtype=np.int64))


def load_sample_records(
    chunks: list[ChunkSpec],
    indices: list[int],
) -> list[SampleRecord]:
    if not indices:
        return []

    offsets = _build_chunk_offsets(chunks)
    total_rows = int(offsets[-1])
    for idx in indices:
        if idx < 0 or idx >= total_rows:
            raise ValueError(f"sample index {idx} out of range [0, {total_rows - 1}]")

    grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for out_pos, global_idx in enumerate(indices):
        chunk_pos = int(np.searchsorted(offsets, global_idx, side="right"))
        chunk_start = 0 if chunk_pos == 0 else int(offsets[chunk_pos - 1])
        chunk_row = global_idx - chunk_start
        grouped[chunk_pos].append((out_pos, global_idx, int(chunk_row)))

    records: list[SampleRecord | None] = [None] * len(indices)
    for chunk_pos, pulls in grouped.items():
        chunk = chunks[chunk_pos]
        pull_rows = np.asarray([item[2] for item in pulls], dtype=np.int64)
        with np.load(chunk.path, mmap_mode="r") as data:
            ox = np.asarray(data["ox"][pull_rows])
            red = np.asarray(data["red"][pull_rows])
            current = np.asarray(data["i"][pull_rows])
            potential = np.asarray(data["e"][pull_rows])
            params = np.asarray(data["p"][pull_rows])

            task_id = _load_optional_ids(data, "task_id", chunk.rows)[pull_rows]
            stage_id = _load_optional_ids(data, "stage_id", chunk.rows)[pull_rows]
            aug_id = _load_optional_ids(data, "aug_id", chunk.rows)[pull_rows]

        for local_idx, (out_pos, global_idx, chunk_row) in enumerate(pulls):
            records[out_pos] = SampleRecord(
                global_index=global_idx,
                chunk_path=chunk.path,
                chunk_row=chunk_row,
                ox=ox[local_idx],
                red=red[local_idx],
                current=current[local_idx],
                potential=potential[local_idx],
                params=params[local_idx],
                task_id=int(task_id[local_idx]),
                stage_id=int(stage_id[local_idx]),
                aug_id=int(aug_id[local_idx]),
            )

    out: list[SampleRecord] = []
    for item in records:
        if item is None:
            raise RuntimeError("Failed to resolve all requested samples")
        out.append(item)
    return out


def summarize_dataset(
    chunks: list[ChunkSpec],
    layout: DatasetLayout,
    labels: LabelNames,
) -> dict[str, Any]:
    total_rows = sum(chunk.rows for chunk in chunks)
    if total_rows <= 0:
        raise ValueError("Dataset appears empty")

    task_counter: Counter[int] = Counter()
    stage_counter: Counter[int] = Counter()
    aug_counter: Counter[int] = Counter()

    nonfinite_rows = 0
    negative_profile_rows = 0
    negative_bulk_rows = 0
    invalid_alpha_rows = 0
    flat_current_rows = 0
    flat_potential_rows = 0

    current_sum = 0.0
    current_sq_sum = 0.0
    current_count = 0
    current_min = float("inf")
    current_max = float("-inf")

    potential_sum = 0.0
    potential_sq_sum = 0.0
    potential_count = 0
    potential_min = float("inf")
    potential_max = float("-inf")

    concentration_min = float("inf")
    concentration_max = float("-inf")

    m = layout.max_species
    for chunk in chunks:
        with np.load(chunk.path, mmap_mode="r") as data:
            ox = np.asarray(data["ox"], dtype=np.float64)
            red = np.asarray(data["red"], dtype=np.float64)
            current = np.asarray(data["i"], dtype=np.float64)
            potential = np.asarray(data["e"], dtype=np.float64)
            params = np.asarray(data["p"], dtype=np.float64)

            rows = ox.shape[0]
            task_id = _load_optional_ids(data, "task_id", rows)
            stage_id = _load_optional_ids(data, "stage_id", rows)
            aug_id = _load_optional_ids(data, "aug_id", rows)

            if rows != chunk.rows:
                raise ValueError(
                    f"Row drift for {chunk.path}: expected {chunk.rows}, got {rows}"
                )

            finite_rows = (
                np.isfinite(ox).all(axis=1)
                & np.isfinite(red).all(axis=1)
                & np.isfinite(current).all(axis=1)
                & np.isfinite(potential).all(axis=1)
                & np.isfinite(params).all(axis=1)
            )
            nonfinite_rows += int(np.sum(~finite_rows))

            ox_row_min = np.min(ox, axis=1)
            red_row_min = np.min(red, axis=1)
            negative_profile_rows += int(
                np.sum((ox_row_min < -1e-6) | (red_row_min < -1e-6))
            )

            c_ox = params[:, 2 * m : 3 * m]
            c_red = params[:, 3 * m : 4 * m]
            negative_bulk_rows += int(np.sum((c_ox < -1e-9).any(axis=1) | (c_red < -1e-9).any(axis=1)))

            alpha = params[:, 6 * m : 7 * m]
            invalid_alpha_rows += int(np.sum((alpha < 0.0).any(axis=1) | (alpha > 1.0).any(axis=1)))

            flat_current_rows += int(np.sum(np.ptp(current, axis=1) < 1e-4))
            flat_potential_rows += int(np.sum(np.ptp(potential, axis=1) < 1e-6))

            unique, counts = np.unique(task_id, return_counts=True)
            task_counter.update({int(u): int(c) for u, c in zip(unique, counts, strict=True)})
            unique, counts = np.unique(stage_id, return_counts=True)
            stage_counter.update({int(u): int(c) for u, c in zip(unique, counts, strict=True)})
            unique, counts = np.unique(aug_id, return_counts=True)
            aug_counter.update({int(u): int(c) for u, c in zip(unique, counts, strict=True)})

            finite_current = current[np.isfinite(current)]
            if finite_current.size:
                current_sum += float(np.sum(finite_current))
                current_sq_sum += float(np.sum(finite_current * finite_current))
                current_count += int(finite_current.size)
                current_min = min(current_min, float(np.min(finite_current)))
                current_max = max(current_max, float(np.max(finite_current)))

            finite_potential = potential[np.isfinite(potential)]
            if finite_potential.size:
                potential_sum += float(np.sum(finite_potential))
                potential_sq_sum += float(np.sum(finite_potential * finite_potential))
                potential_count += int(finite_potential.size)
                potential_min = min(potential_min, float(np.min(finite_potential)))
                potential_max = max(potential_max, float(np.max(finite_potential)))

            finite_conc_ox = ox[np.isfinite(ox)]
            finite_conc_red = red[np.isfinite(red)]
            if finite_conc_ox.size:
                concentration_min = min(concentration_min, float(np.min(finite_conc_ox)))
                concentration_max = max(concentration_max, float(np.max(finite_conc_ox)))
            if finite_conc_red.size:
                concentration_min = min(concentration_min, float(np.min(finite_conc_red)))
                concentration_max = max(concentration_max, float(np.max(finite_conc_red)))

    current_mean = current_sum / current_count if current_count > 0 else float("nan")
    current_var = (
        max(0.0, current_sq_sum / current_count - current_mean * current_mean)
        if current_count > 0
        else float("nan")
    )
    potential_mean = potential_sum / potential_count if potential_count > 0 else float("nan")
    potential_var = (
        max(0.0, potential_sq_sum / potential_count - potential_mean * potential_mean)
        if potential_count > 0
        else float("nan")
    )

    def _counter_table(counter: Counter[int], names: list[str], prefix: str) -> list[dict[str, Any]]:
        max_id = max(counter.keys(), default=-1)
        last = max(max_id, len(names) - 1)
        return [
            {
                "id": idx,
                "name": _name_for_id(names, idx, prefix),
                "count": int(counter.get(idx, 0)),
            }
            for idx in range(last + 1)
        ]

    diagnostics = {
        "nonfinite_rows": nonfinite_rows,
        "negative_profile_rows": negative_profile_rows,
        "negative_bulk_rows": negative_bulk_rows,
        "invalid_alpha_rows": invalid_alpha_rows,
        "flat_current_rows": flat_current_rows,
        "flat_potential_rows": flat_potential_rows,
    }

    def _safe_bound(value: float) -> float:
        return value if np.isfinite(value) else float("nan")

    diagnosis = []
    if nonfinite_rows > 0:
        diagnosis.append("Found non-finite values; dataset has corrupted rows.")
    if negative_profile_rows > 0:
        diagnosis.append("Found negative concentration profile rows.")
    if negative_bulk_rows > 0:
        diagnosis.append("Found negative bulk concentrations in conditioning parameters.")
    if invalid_alpha_rows > 0:
        diagnosis.append("Found alpha values outside [0, 1].")
    if flat_current_rows / total_rows > 0.4:
        diagnosis.append("Large fraction of near-flat current traces (>40%).")
    if flat_potential_rows / total_rows > 0.2:
        diagnosis.append("Large fraction of near-flat potential traces (>20%).")
    if not diagnosis:
        diagnosis.append("No major sanity red flags detected in aggregate checks.")

    return {
        "num_chunks": len(chunks),
        "total_rows": total_rows,
        "layout": {
            "max_species": layout.max_species,
            "nx": layout.nx,
            "signal_len": layout.signal_len,
            "state_width": layout.state_width,
            "phys_dim": layout.phys_dim,
        },
        "ranges": {
            "current_mA": {
                "min": _safe_bound(current_min),
                "max": _safe_bound(current_max),
                "mean": current_mean,
                "std": float(np.sqrt(current_var)),
            },
            "potential_V": {
                "min": _safe_bound(potential_min),
                "max": _safe_bound(potential_max),
                "mean": potential_mean,
                "std": float(np.sqrt(potential_var)),
            },
            "concentration_mM": {
                "min": _safe_bound(concentration_min),
                "max": _safe_bound(concentration_max),
            },
        },
        "diagnostics": diagnostics,
        "distribution": {
            "task": _counter_table(task_counter, labels.task, "task"),
            "stage": _counter_table(stage_counter, labels.stage, "stage"),
            "augmentation": _counter_table(aug_counter, labels.augmentation, "augmentation"),
        },
        "diagnosis": diagnosis,
    }


def _split_params(params: np.ndarray, max_species: int) -> dict[str, np.ndarray]:
    m = max_species
    return {
        "D_ox": np.exp(params[0:m]),
        "D_red": np.exp(params[m : 2 * m]),
        "C_ox": params[2 * m : 3 * m],
        "C_red": params[3 * m : 4 * m],
        "E0": params[4 * m : 5 * m],
        "k0": np.exp(params[5 * m : 6 * m]),
        "alpha": params[6 * m : 7 * m],
    }


def _active_species(param_slices: dict[str, np.ndarray]) -> np.ndarray:
    active = np.where((param_slices["C_ox"] > 1e-6) | (param_slices["C_red"] > 1e-6))[0]
    if active.size == 0:
        return np.asarray([0], dtype=np.int64)
    return active


def _sample_flags(record: SampleRecord) -> list[str]:
    flags: list[str] = []
    if not (
        np.isfinite(record.ox).all()
        and np.isfinite(record.red).all()
        and np.isfinite(record.current).all()
        and np.isfinite(record.potential).all()
        and np.isfinite(record.params).all()
    ):
        flags.append("non_finite")
    if float(np.min(record.ox)) < -1e-6 or float(np.min(record.red)) < -1e-6:
        flags.append("negative_profile")
    if float(np.ptp(record.current)) < 1e-4:
        flags.append("flat_current")
    if float(np.ptp(record.potential)) < 1e-6:
        flags.append("flat_potential")
    return flags


def _augmentation_hint(aug_name: str) -> str:
    if aug_name == "permute_species":
        return "Expected invariant: species order changes while waveform/current should stay consistent."
    if aug_name == "scale_concentration":
        return "Expected invariant: concentrations and current amplitude scale together."
    return "Base sample (no invariant augmentation)."


def _draw_sample(
    fig: Any,
    axes: Any,
    record: SampleRecord,
    layout: DatasetLayout,
    labels: LabelNames,
) -> None:
    ax_e = axes[0, 0]
    ax_i = axes[0, 1]
    ax_cv = axes[1, 0]
    ax_conc = axes[1, 1]
    for ax in (ax_e, ax_i, ax_cv, ax_conc):
        ax.clear()

    t = np.linspace(0.0, 1.0, record.potential.shape[0])
    ax_e.plot(t, record.potential, color="#1f77b4", lw=1.5)
    ax_e.set_title("Applied Potential vs Normalized Time")
    ax_e.set_xlabel("Normalized Time")
    ax_e.set_ylabel("E (V)")

    ax_i.plot(t, record.current, color="#d62728", lw=1.5)
    ax_i.set_title("Current vs Normalized Time")
    ax_i.set_xlabel("Normalized Time")
    ax_i.set_ylabel("I (mA)")

    ax_cv.plot(record.potential, record.current, color="#2ca02c", lw=1.5)
    ax_cv.set_title("I-E Loop (Sample View)")
    ax_cv.set_xlabel("E (V)")
    ax_cv.set_ylabel("I (mA)")

    ox = record.ox.reshape(layout.max_species, layout.nx)
    red = record.red.reshape(layout.max_species, layout.nx)
    x = np.linspace(0.0, 1.0, layout.nx)

    slices = _split_params(record.params, layout.max_species)
    active = _active_species(slices)
    plotted = 0
    for species_idx in active[:4]:
        ax_conc.plot(x, ox[species_idx], lw=1.5, label=f"S{species_idx + 1} ox")
        ax_conc.plot(x, red[species_idx], lw=1.5, ls="--", label=f"S{species_idx + 1} red")
        plotted += 1
    if plotted == 0:
        ax_conc.plot(x, ox[0], lw=1.5, label="S1 ox")
        ax_conc.plot(x, red[0], lw=1.5, ls="--", label="S1 red")
    ax_conc.set_title("Final Concentration Profiles")
    ax_conc.set_xlabel("Normalized Distance")
    ax_conc.set_ylabel("Concentration (mM)")
    ax_conc.legend(loc="best", fontsize=8)

    task_name = _name_for_id(labels.task, record.task_id, "task")
    stage_name = _name_for_id(labels.stage, record.stage_id, "stage")
    aug_name = _name_for_id(labels.augmentation, record.aug_id, "augmentation")
    flags = _sample_flags(record)
    flag_text = "ok" if not flags else ",".join(flags)

    title = (
        f"Sample #{record.global_index} | chunk={record.chunk_path.name}:{record.chunk_row} | "
        f"task={task_name} stage={stage_name} aug={aug_name}"
    )
    subtitle = (
        f"I[min,max,ptp]=({record.current.min():.4g},{record.current.max():.4g},{np.ptp(record.current):.4g}) "
        f"E[min,max,ptp]=({record.potential.min():.4g},{record.potential.max():.4g},{np.ptp(record.potential):.4g}) "
        f"flags={flag_text}"
    )
    fig.suptitle(f"{title}\n{subtitle}\n{_augmentation_hint(aug_name)}", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))


def render_summary_figure(summary: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_task = axes[0, 0]
    ax_stage = axes[0, 1]
    ax_aug = axes[1, 0]
    ax_diag = axes[1, 1]

    def _plot_distribution(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
        filtered = [row for row in rows if int(row["count"]) > 0]
        if not filtered:
            ax.text(0.5, 0.5, "No rows", ha="center", va="center")
            ax.set_title(title)
            return
        labels_local = [str(row["name"]) for row in filtered]
        counts_local = [int(row["count"]) for row in filtered]
        ax.bar(range(len(filtered)), counts_local)
        ax.set_xticks(range(len(filtered)))
        ax.set_xticklabels(labels_local, rotation=35, ha="right")
        ax.set_ylabel("Rows")
        ax.set_title(title)

    _plot_distribution(ax_task, summary["distribution"]["task"], "Task Distribution")
    _plot_distribution(ax_stage, summary["distribution"]["stage"], "Stage Distribution")
    _plot_distribution(ax_aug, summary["distribution"]["augmentation"], "Augmentation Distribution")

    diag = summary["diagnostics"]
    total_rows = max(1, int(summary["total_rows"]))
    diag_items = [
        ("nonfinite", int(diag["nonfinite_rows"])),
        ("neg_profile", int(diag["negative_profile_rows"])),
        ("neg_bulk", int(diag["negative_bulk_rows"])),
        ("invalid_alpha", int(diag["invalid_alpha_rows"])),
        ("flat_current", int(diag["flat_current_rows"])),
        ("flat_potential", int(diag["flat_potential_rows"])),
    ]
    diag_labels = [item[0] for item in diag_items]
    diag_fracs = [100.0 * item[1] / total_rows for item in diag_items]
    ax_diag.bar(range(len(diag_items)), diag_fracs, color="#d62728")
    ax_diag.set_xticks(range(len(diag_items)))
    ax_diag.set_xticklabels(diag_labels, rotation=35, ha="right")
    ax_diag.set_ylabel("Rows (%)")
    ax_diag.set_title("Sanity Flags")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_samples_pdf(
    records: list[SampleRecord],
    layout: DatasetLayout,
    labels: LabelNames,
    out_path: Path,
    n_pages: int,
) -> int:
    if n_pages <= 0 or not records:
        return 0

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pages = 0
    with PdfPages(out_path) as pdf:
        for record in records[:n_pages]:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            _draw_sample(fig, axes, record, layout, labels)
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
    return pages


def launch_interactive_viewer(
    records: list[SampleRecord],
    layout: DatasetLayout,
    labels: LabelNames,
    seed: int,
) -> None:
    if not records:
        print("No sample records available for interactive view.")
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    state = {"idx": 0}
    rng = np.random.default_rng(seed)

    def redraw() -> None:
        _draw_sample(fig, axes, records[state["idx"]], layout, labels)
        fig.canvas.draw_idle()

    def on_key(event: Any) -> None:
        key = str(event.key).lower() if event.key is not None else ""
        if key in {"right", "n", "j"}:
            state["idx"] = (state["idx"] + 1) % len(records)
            redraw()
        elif key in {"left", "p", "k"}:
            state["idx"] = (state["idx"] - 1) % len(records)
            redraw()
        elif key in {"r", "space"}:
            state["idx"] = int(rng.integers(0, len(records)))
            redraw()
        elif key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Interactive controls: n/right=next, p/left=previous, r/space=random, q=quit")
    redraw()
    plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual inspector for ECSFM dataset chunks with random sample browsing and sanity checks.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to dataset directory (chunk_*.npz) or a single .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/ecsfm/dataset_inspector",
        help="Directory to write summary/report artifacts.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for sample selection.")
    parser.add_argument(
        "--n-random",
        type=int,
        default=64,
        help="Number of random rows to load for browsing/gallery when sample-indices is not set.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated global row indices to inspect (overrides n-random).",
    )
    parser.add_argument(
        "--n-gallery",
        type=int,
        default=12,
        help="How many selected samples to render to random_samples.pdf.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive viewer window after reports are generated.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Force headless mode (overrides --show).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    show_viewer = bool(args.show and not args.no_show)
    if not show_viewer:
        import matplotlib

        matplotlib.use("Agg", force=True)

    chunks, layout, labels = scan_dataset(args.dataset)
    total_rows = sum(chunk.rows for chunk in chunks)
    if total_rows <= 0:
        raise ValueError("Dataset contains zero rows.")

    if args.sample_indices is not None:
        selected_indices = parse_global_indices(args.sample_indices, total_rows=total_rows)
    else:
        selected_indices = select_random_global_indices(
            total_rows=total_rows,
            n_samples=max(0, int(args.n_random)),
            seed=args.seed,
        )

    records = load_sample_records(chunks, selected_indices)
    summary = summarize_dataset(chunks, layout, labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "sanity_report.json"
    summary_png = output_dir / "sanity_summary.png"
    samples_pdf = output_dir / "random_samples.pdf"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    render_summary_figure(summary, summary_png)
    pages = render_samples_pdf(records, layout, labels, samples_pdf, n_pages=max(0, int(args.n_gallery)))

    payload = {
        "dataset": str(Path(args.dataset)),
        "num_chunks": summary["num_chunks"],
        "total_rows": summary["total_rows"],
        "selected_rows": len(records),
        "gallery_pages": pages,
        "layout": summary["layout"],
        "diagnostics": summary["diagnostics"],
        "diagnosis": summary["diagnosis"],
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_png": str(summary_png),
            "samples_pdf": str(samples_pdf),
        },
    }
    print(json.dumps(payload, indent=2))

    if show_viewer:
        launch_interactive_viewer(records, layout, labels, seed=args.seed)


if __name__ == "__main__":
    main()
