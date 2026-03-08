"""Canonical test case registry for verification against analytical and COMSOL references."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeciesSpec:
    """Specification for a single electroactive species."""

    name: str
    D: float           # cm^2/s
    z: int             # charge number
    C_bulk: float      # uM


@dataclass(frozen=True)
class KineticsSpec:
    """Butler-Volmer kinetics specification for a redox couple."""

    E0: float           # V
    k0: float           # cm/s
    alpha: float
    n_electrons: int = 1


@dataclass(frozen=True)
class WaveformSpec:
    """Applied potential waveform specification."""

    type: str           # "step", "cv", "eis"
    params: dict = field(default_factory=dict)  # type-specific params


@dataclass(frozen=True)
class TestCaseSpec:
    """Full specification of a verification test case."""

    name: str
    description: str
    category: str                # "diffusion", "migration", "cv", "eis"
    domain_length_cm: float
    n_points: int
    grading_factor: float
    species: list[SpeciesSpec]
    kinetics: list[KineticsSpec]
    waveform: WaveformSpec
    expected_metrics: dict = field(default_factory=dict)
    tolerance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canonical test cases
# ---------------------------------------------------------------------------

CANONICAL_CASES: dict[str, TestCaseSpec] = {}

# 1. Cottrell step - pure diffusion potential step
CANONICAL_CASES["cottrell_step"] = TestCaseSpec(
    name="cottrell_step",
    description=(
        "Pure diffusion potential step: electrode surface concentration driven "
        "to zero. Validates Cottrell equation J = C*sqrt(D/(pi*t)). "
        "Uses simulate_electrochem (implicit diffusion + Butler-Volmer + mass "
        "conservation), so tolerance is wider than the direct Cottrell benchmark."
    ),
    category="diffusion",
    domain_length_cm=0.05,
    n_points=500,
    grading_factor=0.0,
    species=[
        SpeciesSpec(name="Ox", D=1e-5, z=1, C_bulk=1.0),
        SpeciesSpec(name="Red", D=1e-5, z=-1, C_bulk=0.0),
    ],
    kinetics=[
        KineticsSpec(E0=0.0, k0=10.0, alpha=0.5, n_electrons=1),
    ],
    waveform=WaveformSpec(
        type="step",
        params={
            "E_initial": 0.5,
            "E_step": -0.5,
            "t_max": 0.1,
        },
    ),
    expected_metrics={
        "cottrell_rel_error_max": 0.10,
        "cottrell_loglog_slope": -0.5,
    },
    tolerance={
        "cottrell_rel_error_max": 0.10,
        "cottrell_loglog_slope_tol": 0.08,
    },
)

# 2. CV reversible - fast kinetics, Randles-Sevcik validation
CANONICAL_CASES["cv_reversible"] = TestCaseSpec(
    name="cv_reversible",
    description=(
        "Cyclic voltammetry with fast kinetics (k0=0.1 cm/s). "
        "Validates peak separation approaches 59.2/n mV (Nernstian limit) "
        "and peak current follows Randles-Sevcik equation."
    ),
    category="cv",
    domain_length_cm=0.05,
    n_points=200,
    grading_factor=0.0,
    species=[
        SpeciesSpec(name="Ox", D=1e-5, z=1, C_bulk=1.0),
        SpeciesSpec(name="Red", D=1e-5, z=-1, C_bulk=0.0),
    ],
    kinetics=[
        KineticsSpec(E0=0.0, k0=0.1, alpha=0.5, n_electrons=1),
    ],
    waveform=WaveformSpec(
        type="cv",
        params={
            "E_start": 0.5,
            "E_vertex": -0.5,
            "scan_rate": 1.0,
        },
    ),
    expected_metrics={
        "delta_ep_v": 0.0592,
        "randles_sevcik_ip_a_per_cm2": None,  # computed at runtime
    },
    tolerance={
        "delta_ep_abs_tol": 0.025,
        "ip_rel_tol": 0.15,
    },
)

# 3. CV quasi-reversible - COMSOL reference
CANONICAL_CASES["cv_quasireversible"] = TestCaseSpec(
    name="cv_quasireversible",
    description=(
        "Cyclic voltammetry with intermediate kinetics (k0=1e-3 cm/s). "
        "Validates against COMSOL reference data."
    ),
    category="cv",
    domain_length_cm=0.05,
    n_points=200,
    grading_factor=0.0,
    species=[
        SpeciesSpec(name="Ox", D=1e-5, z=1, C_bulk=1.0),
        SpeciesSpec(name="Red", D=1e-5, z=-1, C_bulk=0.0),
    ],
    kinetics=[
        KineticsSpec(E0=0.0, k0=1e-3, alpha=0.5, n_electrons=1),
    ],
    waveform=WaveformSpec(
        type="cv",
        params={
            "E_start": 0.5,
            "E_vertex": -0.5,
            "scan_rate": 1.0,
        },
    ),
    expected_metrics={
        "delta_ep_min_v": 0.10,
    },
    tolerance={
        "comsol_l2_norm": 0.05,
        "comsol_peak_current_rel": 0.10,
    },
)

# 4. CV irreversible - analytical peak shift + COMSOL
CANONICAL_CASES["cv_irreversible"] = TestCaseSpec(
    name="cv_irreversible",
    description=(
        "Cyclic voltammetry with slow kinetics (k0=1e-5 cm/s). "
        "Validates large peak separation and irreversible peak shift behavior."
    ),
    category="cv",
    domain_length_cm=0.05,
    n_points=200,
    grading_factor=0.0,
    species=[
        SpeciesSpec(name="Ox", D=1e-5, z=1, C_bulk=1.0),
        SpeciesSpec(name="Red", D=1e-5, z=-1, C_bulk=0.0),
    ],
    kinetics=[
        KineticsSpec(E0=0.0, k0=1e-5, alpha=0.5, n_electrons=1),
    ],
    waveform=WaveformSpec(
        type="cv",
        params={
            "E_start": 0.5,
            "E_vertex": -0.5,
            "scan_rate": 1.0,
        },
    ),
    expected_metrics={
        "delta_ep_min_v": 0.30,
    },
    tolerance={
        "delta_ep_min_check": 0.30,
        "comsol_l2_norm": 0.05,
    },
)

# 5. Binary electrolyte migration - NP + electroneutrality
CANONICAL_CASES["binary_migration"] = TestCaseSpec(
    name="binary_migration",
    description=(
        "Binary electrolyte (NaCl) with Nernst-Planck transport and "
        "electroneutrality. Validates Henderson equation for junction potential. "
        "Note: current simulator is diffusion-only; this case tests that "
        "diffusion-only results are physically reasonable and serves as a "
        "COMSOL comparison target when migration is implemented."
    ),
    category="migration",
    domain_length_cm=0.05,
    n_points=200,
    grading_factor=0.0,
    species=[
        SpeciesSpec(name="Na+", D=1.334e-5, z=1, C_bulk=100.0),
        SpeciesSpec(name="Cl-", D=2.032e-5, z=-1, C_bulk=100.0),
    ],
    kinetics=[
        KineticsSpec(E0=0.0, k0=0.01, alpha=0.5, n_electrons=1),
    ],
    waveform=WaveformSpec(
        type="step",
        params={
            "E_initial": 0.0,
            "E_step": -0.3,
            "t_max": 0.05,
        },
    ),
    expected_metrics={
        "mass_conserved": True,
    },
    tolerance={
        "mass_conservation_rel": 0.01,
        "comsol_l2_norm": 0.10,
    },
)

# 6. EIS Randles circuit - analytical impedance
CANONICAL_CASES["eis_randles"] = TestCaseSpec(
    name="eis_randles",
    description=(
        "Electrochemical impedance spectroscopy with Randles circuit "
        "(Ru-Cdl-Rct). Validates impedance magnitude and phase against "
        "analytical circuit model."
    ),
    category="eis",
    domain_length_cm=0.05,
    n_points=100,
    grading_factor=0.0,
    species=[
        SpeciesSpec(name="Ox", D=1e-5, z=1, C_bulk=1.0),
        SpeciesSpec(name="Red", D=1e-5, z=-1, C_bulk=0.0),
    ],
    kinetics=[
        KineticsSpec(E0=0.0, k0=0.02, alpha=0.5, n_electrons=1),
    ],
    waveform=WaveformSpec(
        type="eis",
        params={
            "frequencies_hz": [0.5, 1.0, 2.0, 5.0, 10.0],
            "amplitude_v": 0.01,
            "dc_potential_v": -0.02,
            "t_window_s": 8.0,
        },
    ),
    expected_metrics={
        "impedance_trend": "decreasing_magnitude_with_frequency",
    },
    tolerance={
        "z_mag_rel_error": 0.15,
        "z_phase_abs_error_rad": 0.2,
    },
)

# 7. Multi-species CV - 3 species coupled CV
CANONICAL_CASES["multi_species_cv"] = TestCaseSpec(
    name="multi_species_cv",
    description=(
        "Cyclic voltammetry with 3 independent redox couples at different E0 "
        "values. Validates that multiple peaks appear and peak currents scale "
        "correctly. COMSOL reference comparison target."
    ),
    category="cv",
    domain_length_cm=0.05,
    n_points=200,
    grading_factor=0.0,
    species=[
        SpeciesSpec(name="Ox_A", D=1e-5, z=1, C_bulk=1.0),
        SpeciesSpec(name="Red_A", D=1e-5, z=-1, C_bulk=0.0),
        SpeciesSpec(name="Ox_B", D=0.8e-5, z=1, C_bulk=0.5),
        SpeciesSpec(name="Red_B", D=0.8e-5, z=-1, C_bulk=0.0),
        SpeciesSpec(name="Ox_C", D=1.2e-5, z=1, C_bulk=2.0),
        SpeciesSpec(name="Red_C", D=1.2e-5, z=-1, C_bulk=0.0),
    ],
    kinetics=[
        KineticsSpec(E0=0.0, k0=0.01, alpha=0.5, n_electrons=1),
        KineticsSpec(E0=-0.2, k0=0.01, alpha=0.5, n_electrons=1),
        KineticsSpec(E0=0.2, k0=0.01, alpha=0.5, n_electrons=1),
    ],
    waveform=WaveformSpec(
        type="cv",
        params={
            "E_start": 0.6,
            "E_vertex": -0.6,
            "scan_rate": 1.0,
        },
    ),
    expected_metrics={
        "n_species": 3,
    },
    tolerance={
        "comsol_l2_norm": 0.10,
    },
)

# 8. Migration CV - CV with migration effects
CANONICAL_CASES["migration_cv"] = TestCaseSpec(
    name="migration_cv",
    description=(
        "Cyclic voltammetry with migration effects due to low supporting "
        "electrolyte concentration. Validates that migration modifies the "
        "diffusion-only CV shape. Current simulator is diffusion-only; "
        "this case serves as a COMSOL comparison target and documents the "
        "expected deviation when migration is significant."
    ),
    category="migration",
    domain_length_cm=0.05,
    n_points=200,
    grading_factor=0.0,
    species=[
        SpeciesSpec(name="Ox", D=1e-5, z=1, C_bulk=1.0),
        SpeciesSpec(name="Red", D=1e-5, z=-1, C_bulk=0.0),
    ],
    kinetics=[
        KineticsSpec(E0=0.0, k0=0.01, alpha=0.5, n_electrons=1),
    ],
    waveform=WaveformSpec(
        type="cv",
        params={
            "E_start": 0.5,
            "E_vertex": -0.5,
            "scan_rate": 1.0,
        },
    ),
    expected_metrics={
        "migration_effect_qualitative": "peak_shift_and_shape_change",
    },
    tolerance={
        "comsol_l2_norm": 0.15,
    },
)
