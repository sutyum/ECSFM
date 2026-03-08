# COMSOL Reference Data

This directory stores CSV reference data exported from COMSOL Multiphysics
simulations for verification of the ECSFM electrochemical simulator.

## CSV Schema

Each file is named `{case_name}_comsol.csv` and contains:

| Column | Unit | Description |
|--------|------|-------------|
| `time_s` | s | Simulation time |
| `potential_V` | V | Applied or measured potential |
| `current_mA` | mA | Total current response |

## Generating Reference Data

To generate reference data (requires COMSOL + MPh):

```bash
uv run python scripts/run_verification.py --export-comsol-data
```

## Using Cached Data

To run verification against cached COMSOL data without a live COMSOL license:

```bash
uv run python scripts/run_verification.py --offline
```

## Canonical Cases

| Case | Category | Description |
|------|----------|-------------|
| `cottrell_step` | diffusion | Cottrell potential step |
| `cv_reversible` | cv | Reversible CV (k0=0.1) |
| `cv_quasireversible` | cv | Quasi-reversible CV (k0=1e-3) |
| `cv_irreversible` | cv | Irreversible CV (k0=1e-5) |
| `binary_migration` | migration | NaCl binary electrolyte |
| `eis_randles` | eis | Randles circuit EIS |
| `multi_species_cv` | cv | 3-species coupled CV |
| `migration_cv` | migration | CV with migration effects |
