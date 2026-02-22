# AGENTS

Operational runbook for Modal generation/training in this repo.

## Modal GPU status

- `check_gpu` should report CUDA backend and an NVIDIA device (for example A10G).

## Canonical commands (absolute paths)

Generate dataset chunks:

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
```

Train and evaluate:

```bash
uv run modal run /Users/satyamtiwary/Documents/Hardware-Things/ElectrochemicalSensingFM/scripts/modal_pipeline.py::train_model \
  --dataset-name dataset_balanced_742k \
  --artifact-name fullscale_balanced_modal \
  --epochs 18 \
  --batch-size 16 \
  --n-samples 0 \
  --lr 1e-3 \
  --hidden-size 128 \
  --depth 3
```

One-shot full pipeline:

```bash
uv run modal run /Users/satyamtiwary/Documents/Hardware-Things/ElectrochemicalSensingFM/scripts/modal_pipeline.py::full_pipeline \
  --dataset-name dataset_balanced_742k \
  --artifact-name fullscale_balanced_modal \
  --n-chunks 4 \
  --parallelism 4 \
  --n-samples-per-chunk 25000 \
  --epochs 18 \
  --batch-size 16
```
