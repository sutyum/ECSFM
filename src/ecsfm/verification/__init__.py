"""COMSOL Verification System for ECSFM electrochemical simulator.

Provides canonical test cases, automated simulation runner, COMSOL
comparison (optional), quantitative comparator, and report generation.
"""

from ecsfm.verification.test_cases import (
    CANONICAL_CASES,
    KineticsSpec,
    SpeciesSpec,
    TestCaseSpec,
    WaveformSpec,
)
from ecsfm.verification.comparator import ComparisonMetrics, compare_against_analytical, compare_traces
from ecsfm.verification.runner import SimResult, run_case

__all__ = [
    "CANONICAL_CASES",
    "ComparisonMetrics",
    "KineticsSpec",
    "SimResult",
    "SpeciesSpec",
    "TestCaseSpec",
    "WaveformSpec",
    "compare_against_analytical",
    "compare_traces",
    "run_case",
]
