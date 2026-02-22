# ElectrochemicalSensingFM

Physics-driven electrochemical simulation and conditional flow-matching surrogate training.

## Quick Start

```bash
uv sync
uv run pytest -q
```

## Main Workflows

### 1) Generate Dataset

```bash
uv run python src/ecsfm/data/generate.py \
  --n-samples 1000 \
  --n-chunks 1 \
  --max-species 5 \
  --backend auto \
  --sim-steps 512 \
  --device-batch-size 128 \
  --recipe curriculum_multitask \
  --output-dir /tmp/ecsfm/dataset_massive
```

Outputs chunked `.npz` files with keys:
- `ox`: flattened final oxidized concentration profile
- `red`: flattened final reduced concentration profile
- `i`: resampled measured current trace
- `e`: resampled applied potential trace
- `p`: flattened physical conditioning parameters
- `task_id`: task label index used for multitask conditioning
- `stage_id`: curriculum stage label index
- `aug_id`: augmentation label index (`0` means no augmentation)
- `task_names`, `stage_names`, `augmentation_names`: label vocabularies

Useful generation flags:
- `--recipe {baseline_random,curriculum_multitask,stress_mixture}`
- `--stage-proportions foundation,bridge,frontier` (curriculum recipe)
- `--invariant-fraction 0.35` to add invariant pair augmentations
- `--no-invariants` to disable augmentation pairs
- `--backend {auto,process_pool,gpu_batch}` (`auto` uses `gpu_batch` on GPU/TPU)
- `--sim-steps 512` controls fixed integration steps in `gpu_batch`
- `--device-batch-size 128` controls GPU kernel batch size per launch
- `--progress {auto,on,off}` controls tqdm display; use `off` for clean parallel logs

### 2) Train Surrogate

```bash
uv run python -m ecsfm.fm.train \
  --dataset /tmp/ecsfm/dataset_massive \
  --artifact-dir /tmp/ecsfm \
  --epochs 500 \
  --batch-size 32 \
  --curriculum
```

Use `--no-curriculum` to disable stage-based sampling.

Training artifacts are written to `--artifact-dir`:
- `surrogate_model.eqx`
- `training_history.json`
- `loss_curve.png`
- `surrogate_comparison_ep*.png`
- `normalizers.npz`
- `model_meta.json`

Optional config file:

```bash
uv run python -m ecsfm.fm.train --config config.json --dataset /tmp/ecsfm/dataset_massive
```

### 3) Evaluate Surrogate

```bash
uv run python -m ecsfm.fm.eval_classical \
  --checkpoint /tmp/ecsfm/surrogate_model.eqx \
  --output-dir /tmp/ecsfm
```

Produces per-scenario comparison plots and `evaluation_scorecard.json`.
If `normalizers.npz` and `model_meta.json` are present beside the checkpoint, evaluation uses them automatically. If they are missing, pass `--dataset`.
Scorecard now includes:
- `Final_Score_Out_Of_100`: robust composite score (nRMSE, nMAE, peak error, correlation)
- `Legacy_R2_Score_Out_Of_100`: previous R2-based score for backward comparison

### 4) Verify Convenience Script

```bash
uv run python scripts/verify_surrogate.py \
  --checkpoint /tmp/ecsfm/surrogate_model.eqx
```

### 5) Hyperparameter Tuning

```bash
uv run python scripts/tune_surrogate.py \
  --output-root tune_runs \
  --recipes curriculum_multitask,stress_mixture \
  --invariant-fractions 0.0,0.35 \
  --learning-rates 1e-3,5e-4 \
  --depths 3,4 \
  --hidden-sizes 128,192
```

Writes under `tune_runs/`:
- `tuning_summary.json`: full ranked run table
- `recommended_config.json`: best-run recommendation bundle
- best training config at `--write-config` (default `config.json`)

### 6) Run Experimental Scenarios

```bash
uv run python scripts/run_experimental_scenarios.py \
  --output-root /tmp/ecsfm/experiments \
  --n-samples 64 \
  --epochs 120
```

This runs three scenarios (baseline, curriculum, curriculum+invariants) and writes:
- `/tmp/ecsfm/experiments/scenario_summary.json`
- `/tmp/ecsfm/experiments/scenario_report.md`

### 7) Scale Out On Modal (GPU)

Install dependencies and authenticate once:

```bash
uv sync
uv run modal setup
```

Sanity check GPU runtime:

```bash
uv run modal run scripts/modal_pipeline.py::check_gpu
```

Generate chunks in parallel on Modal (GPU-backed workers):

```bash
uv run modal run scripts/modal_pipeline.py::generate_dataset \
  --dataset-name dataset_balanced_742k \
  --n-chunks 4 \
  --parallelism 4 \
  --n-samples-per-chunk 25000 \
  --max-species 5 \
  --nx 24 \
  --workers 1 \
  --seed 2026 \
  --recipe curriculum_multitask \
  --invariant-fraction 0.35
```

Train on Modal GPU and run evaluation:

```bash
uv run modal run scripts/modal_pipeline.py::train_model \
  --dataset-name dataset_balanced_742k \
  --artifact-name fullscale_balanced_modal \
  --epochs 18 \
  --batch-size 16 \
  --n-samples 0 \
  --lr 1e-3 \
  --hidden-size 128 \
  --depth 3
```

One command for generation + training + evaluation:

```bash
uv run modal run scripts/modal_pipeline.py::full_pipeline \
  --dataset-name dataset_balanced_742k \
  --artifact-name fullscale_balanced_modal \
  --n-chunks 4 \
  --parallelism 4 \
  --n-samples-per-chunk 25000 \
  --epochs 18 \
  --batch-size 16
```

Exact absolute-path variants on this machine:

```bash
uv run modal run /Users/satyamtiwary/Documents/Hardware-Things/ElectrochemicalSensingFM/scripts/modal_pipeline.py::generate_dataset \
  --dataset-name dataset_balanced_742k \
  --n-chunks 4 \
  --parallelism 4 \
  --n-samples-per-chunk 25000 \
  --max-species 5 \
  --nx 24 \
  --workers 1 \
  --seed 2026 \
  --recipe curriculum_multitask \
  --invariant-fraction 0.35

uv run modal run /Users/satyamtiwary/Documents/Hardware-Things/ElectrochemicalSensingFM/scripts/modal_pipeline.py::train_model \
  --dataset-name dataset_balanced_742k \
  --artifact-name fullscale_balanced_modal \
  --epochs 18 \
  --batch-size 16 \
  --n-samples 0 \
  --lr 1e-3 \
  --hidden-size 128 \
  --depth 3

uv run modal run /Users/satyamtiwary/Documents/Hardware-Things/ElectrochemicalSensingFM/scripts/modal_pipeline.py::full_pipeline \
  --dataset-name dataset_balanced_742k \
  --artifact-name fullscale_balanced_modal \
  --n-chunks 4 \
  --parallelism 4 \
  --n-samples-per-chunk 25000 \
  --epochs 18 \
  --batch-size 16
```

Modal volumes used by this workflow:
- datasets: `ecsfm-datasets` mounted at `/vol/datasets`
- artifacts: `ecsfm-artifacts` mounted at `/vol/artifacts`

### 8) Visual Dataset Inspector

Run local sanity summary + random sample gallery:

```bash
uv run python -m ecsfm.data.inspect \
  --dataset /tmp/ecsfm/dataset_massive \
  --output-dir /tmp/ecsfm/inspect \
  --n-random 64 \
  --n-gallery 12
```

Open an interactive browser over the selected random rows:

```bash
uv run python -m ecsfm.data.inspect \
  --dataset /tmp/ecsfm/dataset_massive \
  --output-dir /tmp/ecsfm/inspect \
  --n-random 64 \
  --show
```

Interactive controls:
- `n`/right: next sample
- `p`/left: previous sample
- `r`/space: jump to random sample
- `q`: close

Inspector artifacts:
- `sanity_report.json`: aggregate diagnostics and label distributions
- `sanity_summary.png`: distribution and sanity-flag charts
- `random_samples.pdf`: per-sample visual pages for quick review

Inspect a dataset stored in Modal volume:

```bash
uv run modal run scripts/modal_pipeline.py::inspect_dataset \
  --dataset-name dataset_balanced_742k \
  --report-name dataset_balanced_742k_inspection \
  --n-random 96 \
  --n-gallery 24
```

### 9) Canonical Electrochem Benchmarks

Run the expert-style benchmark test battery:

```bash
uv run pytest -q tests/test_echem_benchmarks.py
```

Generate a benchmark report with plots:

```bash
uv run python scripts/report_echem_benchmarks.py \
  --output-dir /tmp/ecsfm/echem_benchmarks
```

Artifacts:
- `/tmp/ecsfm/echem_benchmarks/benchmark_summary.json`
- `/tmp/ecsfm/echem_benchmarks/benchmark_report.md`
- `/tmp/ecsfm/echem_benchmarks/canonical_cv_cottrell.png`
- `/tmp/ecsfm/echem_benchmarks/sensor_bode.png`

## Interactive CV Playground

```bash
uv run python scripts/cv_playground.py --scan-rate 0.1 --nx 200
```

## Notes

- Simulators validate key inputs (grid size, positivity constraints, waveform validity).
- Concentration histories are sampled during simulation to keep memory usage bounded.
- For reproducibility, keep dataset generation, training, and evaluation on the same parameter layout (`max-species`, `nx`, signal length).
