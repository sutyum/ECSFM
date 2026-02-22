from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "ecsfm-gpu-scale"
DATASETS_VOLUME_NAME = "ecsfm-datasets"
ARTIFACTS_VOLUME_NAME = "ecsfm-artifacts"
DATASETS_MOUNT = Path("/vol/datasets")
ARTIFACTS_MOUNT = Path("/vol/artifacts")

app = modal.App(APP_NAME)

datasets_volume = modal.Volume.from_name(DATASETS_VOLUME_NAME, create_if_missing=True)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("jax[cuda12]>=0.4.30")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORMS": "cuda,cpu",
        }
    )
    .add_local_dir("src", "/root/src")
)


def _safe_subpath(name: str) -> str:
    cleaned = name.strip().strip("/")
    if not cleaned:
        raise ValueError("Path segment must be non-empty")

    parts = Path(cleaned).parts
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"Illegal path traversal in '{name}'")
    return cleaned


def _run(cmd: list[str]) -> None:
    printable = " ".join(shlex.quote(arg) for arg in cmd)
    print(f"$ {printable}", flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=image, gpu="A10G", timeout=60 * 15)
def probe_gpu() -> dict[str, Any]:
    import jax

    return {
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "platform_version": jax.devices()[0].device_kind if jax.devices() else "unknown",
    }


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 12,
    volumes={str(DATASETS_MOUNT): datasets_volume},
)
def generate_chunk_remote(
    dataset_subdir: str,
    chunk_idx: int,
    n_samples: int = 25_000,
    max_species: int = 5,
    nx: int = 24,
    workers: int = 1,
    seed: int = 2026,
    recipe: str = "curriculum_multitask",
    stage_proportions: str = "0.4,0.35,0.25",
    invariant_fraction: float = 0.35,
    target_sig_len: int = 200,
    no_invariants: bool = False,
) -> dict[str, Any]:
    import numpy as np

    dataset_dir = DATASETS_MOUNT / _safe_subpath(dataset_subdir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "ecsfm.data.generate",
        "--n-samples",
        str(n_samples),
        "--n-chunks",
        "1",
        "--start-chunk",
        str(chunk_idx),
        "--max-species",
        str(max_species),
        "--nx",
        str(nx),
        "--workers",
        str(workers),
        "--seed",
        str(seed),
        "--recipe",
        recipe,
        "--stage-proportions",
        stage_proportions,
        "--target-sig-len",
        str(target_sig_len),
        "--invariant-fraction",
        str(invariant_fraction),
        "--progress",
        "off",
        "--output-dir",
        str(dataset_dir),
    ]
    if no_invariants:
        cmd.append("--no-invariants")

    _run(cmd)

    chunk_path = dataset_dir / f"chunk_{chunk_idx}.npz"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Expected chunk output at {chunk_path}")

    with np.load(chunk_path) as data:
        rows = int(data["ox"].shape[0])

    datasets_volume.commit()

    return {
        "dataset_subdir": dataset_subdir,
        "chunk_idx": chunk_idx,
        "rows": rows,
        "chunk_path": str(chunk_path),
    }


@app.function(
    image=image,
    timeout=60 * 10,
    volumes={str(DATASETS_MOUNT): datasets_volume},
)
def next_chunk_index_remote(dataset_subdir: str) -> int:
    from ecsfm.data.generate import infer_next_chunk_index

    dataset_dir = DATASETS_MOUNT / _safe_subpath(dataset_subdir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return int(infer_next_chunk_index(str(dataset_dir)))


@app.function(
    image=image,
    timeout=60 * 10,
    volumes={str(DATASETS_MOUNT): datasets_volume},
)
def summarize_dataset_remote(dataset_subdir: str) -> dict[str, Any]:
    import numpy as np

    dataset_dir = DATASETS_MOUNT / _safe_subpath(dataset_subdir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(dataset_dir.glob("chunk_*.npz"))
    total_rows = 0
    chunk_ids: list[int] = []
    for file in chunk_files:
        with np.load(file) as data:
            total_rows += int(data["ox"].shape[0])
        stem = file.stem
        try:
            chunk_ids.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue

    return {
        "dataset_subdir": dataset_subdir,
        "dataset_path": str(dataset_dir),
        "num_chunks": len(chunk_files),
        "chunk_ids": sorted(chunk_ids),
        "total_rows": total_rows,
    }


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={
        str(DATASETS_MOUNT): datasets_volume,
        str(ARTIFACTS_MOUNT): artifacts_volume,
    },
)
def inspect_dataset_remote(
    dataset_subdir: str,
    report_subdir: str = "dataset_inspection",
    seed: int = 2026,
    n_random: int = 64,
    n_gallery: int = 12,
    sample_indices: str = "",
) -> dict[str, Any]:
    datasets_volume.reload()

    dataset_dir = DATASETS_MOUNT / _safe_subpath(dataset_subdir)
    report_dir = ARTIFACTS_MOUNT / _safe_subpath(report_subdir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "ecsfm.data.inspect",
        "--dataset",
        str(dataset_dir),
        "--output-dir",
        str(report_dir),
        "--seed",
        str(seed),
        "--n-random",
        str(n_random),
        "--n-gallery",
        str(n_gallery),
        "--no-show",
    ]
    sample_indices = sample_indices.strip()
    if sample_indices:
        cmd.extend(["--sample-indices", sample_indices])

    _run(cmd)

    summary_path = report_dir / "sanity_report.json"
    summary_png = report_dir / "sanity_summary.png"
    samples_pdf = report_dir / "random_samples.pdf"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing inspector report at {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    artifacts_volume.commit()

    return {
        "dataset_subdir": dataset_subdir,
        "report_subdir": report_subdir,
        "report_dir": str(report_dir),
        "total_rows": int(summary.get("total_rows", 0)),
        "num_chunks": int(summary.get("num_chunks", 0)),
        "diagnosis": summary.get("diagnosis", []),
        "artifacts": {
            "summary_json": str(summary_path),
            "summary_png": str(summary_png),
            "samples_pdf": str(samples_pdf),
        },
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 24,
    volumes={
        str(DATASETS_MOUNT): datasets_volume,
        str(ARTIFACTS_MOUNT): artifacts_volume,
    },
)
def train_remote(
    dataset_subdir: str,
    artifact_subdir: str,
    epochs: int = 18,
    batch_size: int = 16,
    n_samples: int = 0,
    lr: float = 1e-3,
    hidden_size: int = 128,
    depth: int = 3,
    seed: int = 2026,
    val_split: float = 0.2,
    curriculum: bool = False,
    new_run: bool = True,
) -> dict[str, Any]:
    datasets_volume.reload()

    dataset_dir = DATASETS_MOUNT / _safe_subpath(dataset_subdir)
    artifact_dir = ARTIFACTS_MOUNT / _safe_subpath(artifact_subdir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "ecsfm.fm.train",
        "--dataset",
        str(dataset_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--n-samples",
        str(n_samples),
        "--lr",
        str(lr),
        "--hidden-size",
        str(hidden_size),
        "--depth",
        str(depth),
        "--seed",
        str(seed),
        "--val-split",
        str(val_split),
    ]
    cmd.append("--curriculum" if curriculum else "--no-curriculum")
    cmd.append("--new-run" if new_run else "--resume")

    _run(cmd)

    history_path = artifact_dir / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing training history at {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history_payload = json.load(f)

    history = history_payload.get("history", {})
    train_hist = history.get("train", [])
    val_hist = history.get("val", [])

    artifacts_volume.commit()

    return {
        "dataset_subdir": dataset_subdir,
        "artifact_subdir": artifact_subdir,
        "artifact_dir": str(artifact_dir),
        "epochs": epochs,
        "final_train_loss": float(train_hist[-1]) if train_hist else None,
        "best_val_loss": min((float(v[1]) for v in val_hist), default=None),
        "num_train_points": len(train_hist),
        "num_val_points": len(val_hist),
    }


@app.function(
    image=image,
    timeout=60 * 60 * 4,
    volumes={str(ARTIFACTS_MOUNT): artifacts_volume},
)
def evaluate_remote(
    artifact_subdir: str,
    eval_subdir: str = "eval",
    seed: int = 2026,
) -> dict[str, Any]:
    artifacts_volume.reload()

    artifact_dir = ARTIFACTS_MOUNT / _safe_subpath(artifact_subdir)
    eval_dir = artifact_dir / _safe_subpath(eval_subdir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = artifact_dir / "surrogate_model.eqx"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint}")

    cmd = [
        sys.executable,
        "-m",
        "ecsfm.fm.eval_classical",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(eval_dir),
        "--seed",
        str(seed),
    ]
    _run(cmd)

    scorecard_path = eval_dir / "evaluation_scorecard.json"
    if not scorecard_path.exists():
        raise FileNotFoundError(f"Missing scorecard at {scorecard_path}")

    with open(scorecard_path, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    artifacts_volume.commit()

    return {
        "artifact_subdir": artifact_subdir,
        "eval_subdir": eval_subdir,
        "scorecard_path": str(scorecard_path),
        "final_score": float(scorecard.get("Final_Score_Out_Of_100", 0.0)),
        "legacy_r2_score": float(scorecard.get("Legacy_R2_Score_Out_Of_100", 0.0)),
    }


def _collect_spawn_result(
    pending: list[tuple[int, modal.FunctionCall]],
    results: list[dict[str, Any]],
) -> None:
    chunk_idx, function_call = pending.pop(0)
    output = function_call.get()
    results.append(output)
    print(
        f"Finished chunk {chunk_idx}: rows={output['rows']} "
        f"({output['chunk_path']})",
        flush=True,
    )


def _run_generation_plan(
    dataset_name: str,
    n_chunks: int,
    start_chunk: int,
    parallelism: int,
    n_samples_per_chunk: int,
    max_species: int,
    nx: int,
    workers: int,
    seed: int,
    recipe: str,
    stage_proportions: str,
    invariant_fraction: float,
    target_sig_len: int,
    no_invariants: bool,
) -> dict[str, Any]:
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be > 0, got {n_chunks}")
    if parallelism <= 0:
        raise ValueError(f"parallelism must be > 0, got {parallelism}")

    if start_chunk < 0:
        start_chunk = int(next_chunk_index_remote.remote(dataset_name))
        print(f"Auto-detected start_chunk={start_chunk}", flush=True)

    chunk_ids = list(range(start_chunk, start_chunk + n_chunks))
    pending: list[tuple[int, modal.FunctionCall]] = []
    results: list[dict[str, Any]] = []

    for chunk_idx in chunk_ids:
        fn_call = generate_chunk_remote.spawn(
            dataset_subdir=dataset_name,
            chunk_idx=chunk_idx,
            n_samples=n_samples_per_chunk,
            max_species=max_species,
            nx=nx,
            workers=workers,
            seed=seed,
            recipe=recipe,
            stage_proportions=stage_proportions,
            invariant_fraction=invariant_fraction,
            target_sig_len=target_sig_len,
            no_invariants=no_invariants,
        )
        pending.append((chunk_idx, fn_call))
        if len(pending) >= parallelism:
            _collect_spawn_result(pending, results)

    while pending:
        _collect_spawn_result(pending, results)

    dataset_summary = summarize_dataset_remote.remote(dataset_name)
    return {
        "generated_chunks": sorted(results, key=lambda item: int(item["chunk_idx"])),
        "dataset_summary": dataset_summary,
    }


@app.local_entrypoint()
def check_gpu() -> None:
    details = probe_gpu.remote()
    print(json.dumps(details, indent=2))


@app.local_entrypoint()
def generate_dataset(
    dataset_name: str = "dataset_balanced_742k_modal",
    n_chunks: int = 4,
    start_chunk: int = -1,
    parallelism: int = 4,
    n_samples_per_chunk: int = 25_000,
    max_species: int = 5,
    nx: int = 24,
    workers: int = 1,
    seed: int = 2026,
    recipe: str = "curriculum_multitask",
    stage_proportions: str = "0.4,0.35,0.25",
    invariant_fraction: float = 0.35,
    target_sig_len: int = 200,
    no_invariants: bool = False,
) -> None:
    payload = _run_generation_plan(
        dataset_name=dataset_name,
        n_chunks=n_chunks,
        start_chunk=start_chunk,
        parallelism=parallelism,
        n_samples_per_chunk=n_samples_per_chunk,
        max_species=max_species,
        nx=nx,
        workers=workers,
        seed=seed,
        recipe=recipe,
        stage_proportions=stage_proportions,
        invariant_fraction=invariant_fraction,
        target_sig_len=target_sig_len,
        no_invariants=no_invariants,
    )
    print(json.dumps(payload, indent=2))


@app.local_entrypoint()
def inspect_dataset(
    dataset_name: str = "dataset_balanced_742k",
    report_name: str = "dataset_balanced_742k_inspection",
    seed: int = 2026,
    n_random: int = 64,
    n_gallery: int = 12,
    sample_indices: str = "",
) -> None:
    payload = inspect_dataset_remote.remote(
        dataset_subdir=dataset_name,
        report_subdir=report_name,
        seed=seed,
        n_random=n_random,
        n_gallery=n_gallery,
        sample_indices=sample_indices,
    )
    print(json.dumps(payload, indent=2))


@app.local_entrypoint()
def train_model(
    dataset_name: str = "dataset_balanced_742k_modal",
    artifact_name: str = "fullscale_balanced_modal",
    epochs: int = 18,
    batch_size: int = 16,
    n_samples: int = 0,
    lr: float = 1e-3,
    hidden_size: int = 128,
    depth: int = 3,
    seed: int = 2026,
    val_split: float = 0.2,
    curriculum: bool = False,
    new_run: bool = True,
    run_eval: bool = True,
    eval_subdir: str = "eval",
) -> None:
    train_result = train_remote.remote(
        dataset_subdir=dataset_name,
        artifact_subdir=artifact_name,
        epochs=epochs,
        batch_size=batch_size,
        n_samples=n_samples,
        lr=lr,
        hidden_size=hidden_size,
        depth=depth,
        seed=seed,
        val_split=val_split,
        curriculum=curriculum,
        new_run=new_run,
    )
    payload: dict[str, Any] = {"train": train_result}

    if run_eval:
        eval_result = evaluate_remote.remote(
            artifact_subdir=artifact_name,
            eval_subdir=eval_subdir,
            seed=seed,
        )
        payload["eval"] = eval_result

    print(json.dumps(payload, indent=2))


@app.local_entrypoint()
def full_pipeline(
    dataset_name: str = "dataset_balanced_742k_modal",
    artifact_name: str = "fullscale_balanced_modal",
    n_chunks: int = 4,
    start_chunk: int = -1,
    parallelism: int = 4,
    n_samples_per_chunk: int = 25_000,
    max_species: int = 5,
    nx: int = 24,
    workers: int = 1,
    seed: int = 2026,
    recipe: str = "curriculum_multitask",
    stage_proportions: str = "0.4,0.35,0.25",
    invariant_fraction: float = 0.35,
    target_sig_len: int = 200,
    no_invariants: bool = False,
    epochs: int = 18,
    batch_size: int = 16,
    n_samples: int = 0,
    lr: float = 1e-3,
    hidden_size: int = 128,
    depth: int = 3,
    val_split: float = 0.2,
    curriculum: bool = False,
    new_run: bool = True,
    eval_subdir: str = "eval",
) -> None:
    generation_payload = _run_generation_plan(
        dataset_name=dataset_name,
        n_chunks=n_chunks,
        start_chunk=start_chunk,
        parallelism=parallelism,
        n_samples_per_chunk=n_samples_per_chunk,
        max_species=max_species,
        nx=nx,
        workers=workers,
        seed=seed,
        recipe=recipe,
        stage_proportions=stage_proportions,
        invariant_fraction=invariant_fraction,
        target_sig_len=target_sig_len,
        no_invariants=no_invariants,
    )
    print(json.dumps(generation_payload, indent=2))

    train_result = train_remote.remote(
        dataset_subdir=dataset_name,
        artifact_subdir=artifact_name,
        epochs=epochs,
        batch_size=batch_size,
        n_samples=n_samples,
        lr=lr,
        hidden_size=hidden_size,
        depth=depth,
        seed=seed,
        val_split=val_split,
        curriculum=curriculum,
        new_run=new_run,
    )
    eval_result = evaluate_remote.remote(
        artifact_subdir=artifact_name,
        eval_subdir=eval_subdir,
        seed=seed,
    )

    print(
        json.dumps(
            {
                "train": train_result,
                "eval": eval_result,
            },
            indent=2,
        )
    )
