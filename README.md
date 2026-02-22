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
  --output-dir /tmp/ecsfm/dataset_massive
```

Outputs chunked `.npz` files with keys:
- `ox`: flattened final oxidized concentration profile
- `red`: flattened final reduced concentration profile
- `i`: resampled measured current trace
- `e`: resampled applied potential trace
- `p`: flattened physical conditioning parameters

### 2) Train Surrogate

```bash
uv run python -m ecsfm.fm.train \
  --dataset /tmp/ecsfm/dataset_massive \
  --artifact-dir /tmp/ecsfm \
  --epochs 500 \
  --batch-size 32
```

Training artifacts are written to `--artifact-dir`:
- `surrogate_model.eqx`
- `training_history.json`
- `loss_curve.png`
- `surrogate_comparison_ep*.png`

Optional config file:

```bash
uv run python -m ecsfm.fm.train --config config.json --dataset /tmp/ecsfm/dataset_massive
```

### 3) Evaluate Surrogate

```bash
uv run python -m ecsfm.fm.eval_classical \
  --checkpoint /tmp/ecsfm/surrogate_model.eqx \
  --dataset /tmp/ecsfm/dataset_massive \
  --output-dir /tmp/ecsfm
```

Produces per-scenario comparison plots and `evaluation_scorecard.json`.

### 4) Verify Convenience Script

```bash
uv run python scripts/verify_surrogate.py \
  --checkpoint /tmp/ecsfm/surrogate_model.eqx \
  --dataset /tmp/ecsfm/dataset_massive
```

### 5) Hyperparameter Tuning

```bash
uv run python scripts/tune_surrogate.py \
  --dataset /tmp/ecsfm/dataset_massive \
  --artifact-root tune_runs
```

Writes run-wise artifacts under `tune_runs/` and updates `config.json` with the best run.

## Interactive CV Playground

```bash
uv run python scripts/cv_playground.py --scan-rate 0.1 --nx 200
```

## Notes

- Simulators validate key inputs (grid size, positivity constraints, waveform validity).
- Concentration histories are sampled during simulation to keep memory usage bounded.
- For reproducibility, keep dataset generation, training, and evaluation on the same parameter layout (`max-species`, `nx`, signal length).
