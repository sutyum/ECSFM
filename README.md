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

## Interactive CV Playground

```bash
uv run python scripts/cv_playground.py --scan-rate 0.1 --nx 200
```

## Notes

- Simulators validate key inputs (grid size, positivity constraints, waveform validity).
- Concentration histories are sampled during simulation to keep memory usage bounded.
- For reproducibility, keep dataset generation, training, and evaluation on the same parameter layout (`max-species`, `nx`, signal length).
