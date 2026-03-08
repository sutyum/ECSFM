"""COMSOL automation via MPh for verification reference data generation.

All functions gracefully handle the case where MPh (and COMSOL) is not installed,
printing a warning and returning None.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ecsfm.verification.test_cases import TestCaseSpec

logger = logging.getLogger(__name__)

_MPH_AVAILABLE = False
try:
    import mph  # noqa: F401

    _MPH_AVAILABLE = True
except ImportError:
    pass


def is_comsol_available() -> bool:
    """Return True if MPh and COMSOL are usable."""
    if not _MPH_AVAILABLE:
        return False
    try:
        client = mph.start()
        client.disconnect()
        return True
    except Exception:
        return False


def _warn_no_comsol(action: str) -> None:
    msg = (
        f"Cannot {action}: MPh/COMSOL not available. "
        "Install with `pip install mph` and ensure COMSOL is licensed."
    )
    warnings.warn(msg, stacklevel=3)
    logger.warning(msg)


def build_comsol_model(spec: TestCaseSpec) -> Any | None:
    """Build a COMSOL model from a test case spec using MPh.

    Returns the mph model object, or None if COMSOL is not available.
    """
    if not _MPH_AVAILABLE:
        _warn_no_comsol("build COMSOL model")
        return None

    import mph

    client = mph.start()
    model = client.create(spec.name)

    # ---- Geometry: 1D interval [0, L] ----
    geom = model / "geometries" / "Geometry 1"
    geom.java.create("i1", "Interval")
    geom.java.feature("i1").set("p1", 0.0)
    geom.java.feature("i1").set("p2", spec.domain_length_cm * 1e-2)  # convert cm to m
    geom.java.run("fin")

    # ---- Parameters ----
    params = model / "parameters"
    n_kinetics = len(spec.kinetics)

    for i in range(n_kinetics):
        ox_idx = 2 * i
        red_idx = 2 * i + 1
        kin = spec.kinetics[i]
        if ox_idx < len(spec.species):
            ox = spec.species[ox_idx]
            params.java.set(f"D_ox_{i}", f"{ox.D} [cm^2/s]")
            params.java.set(f"C_bulk_ox_{i}", f"{ox.C_bulk * 1e-3} [mol/m^3]")
        if red_idx < len(spec.species):
            red = spec.species[red_idx]
            params.java.set(f"D_red_{i}", f"{red.D} [cm^2/s]")
            params.java.set(f"C_bulk_red_{i}", f"{red.C_bulk * 1e-3} [mol/m^3]")

        params.java.set(f"E0_{i}", f"{kin.E0} [V]")
        params.java.set(f"k0_{i}", f"{kin.k0} [cm/s]")
        params.java.set(f"alpha_{i}", str(kin.alpha))

    # ---- Physics: Transport of Diluted Species ----
    physics = model / "physics"
    tds = physics.java.create("tds", "DilutedSpecies", "geom1")

    for i in range(n_kinetics):
        sp_name = f"sp{i}"
        tds.java.create(sp_name, "Species")
        tds.java.feature(sp_name).set("D", f"D_ox_{i}")

    # ---- Mesh ----
    mesh = model / "meshes" / "Mesh 1"
    mesh.java.create("edg1", "Edge")
    mesh.java.feature("edg1").create("size1", "Size")
    mesh.java.feature("edg1").feature("size1").set("custom", True)
    mesh.java.feature("edg1").feature("size1").set(
        "hmax", str(spec.domain_length_cm * 1e-2 / spec.n_points)
    )
    mesh.java.run()

    # ---- Study ----
    study = model / "studies"
    std = study.java.create("std1", "Study")

    wf = spec.waveform
    if wf.type in ("step", "cv"):
        t_max = wf.params.get("t_max", 1.0)
        std.java.create("time", "Transient")
        std.java.feature("time").set("tlist", f"range(0, {t_max / 200}, {t_max})")
    elif wf.type == "eis":
        freqs = wf.params.get("frequencies_hz", [1.0])
        freq_str = " ".join(str(f) for f in freqs)
        std.java.create("freq", "Frequency")
        std.java.feature("freq").set("plist", freq_str)

    logger.info("COMSOL model built for case '%s'", spec.name)
    return model


def run_comsol_case(spec: TestCaseSpec) -> dict[str, np.ndarray] | None:
    """Build, solve, and extract results from COMSOL.

    Returns a dict with time/potential/current arrays, or None if COMSOL
    is not available.
    """
    if not _MPH_AVAILABLE:
        _warn_no_comsol("run COMSOL case")
        return None

    model = build_comsol_model(spec)
    if model is None:
        return None

    try:
        model.solve()

        # Extract time series from the first evaluation point (electrode surface)
        t_data = np.array(model.evaluate("t"))
        E_data = np.array(model.evaluate("E_applied"))
        I_data = np.array(model.evaluate("I_total"))

        return {
            "time": t_data,
            "potential": E_data,
            "current": I_data,
        }
    except Exception as exc:
        logger.error("COMSOL solve failed for '%s': %s", spec.name, exc)
        return None


def export_reference_data(
    spec: TestCaseSpec,
    output_dir: str | Path,
) -> Path | None:
    """Run COMSOL and export CSV reference data.

    Returns the path to the exported CSV, or None if COMSOL is not available.
    """
    if not _MPH_AVAILABLE:
        _warn_no_comsol("export COMSOL reference data")
        return None

    result = run_comsol_case(spec)
    if result is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{spec.name}_comsol.csv"

    header = "time_s,potential_V,current_mA"
    data = np.column_stack([result["time"], result["potential"], result["current"]])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    logger.info("Exported COMSOL reference data to %s", csv_path)
    return csv_path


def load_reference_data(
    case_name: str,
    data_dir: str | Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Load previously exported COMSOL reference CSV data.

    Returns a dict with time/potential/current arrays, or None if the file
    does not exist.
    """
    if data_dir is None:
        # Default to data/comsol_reference/ relative to project root
        data_dir = Path(__file__).resolve().parents[3] / "data" / "comsol_reference"
    else:
        data_dir = Path(data_dir)

    csv_path = data_dir / f"{case_name}_comsol.csv"
    if not csv_path.exists():
        logger.debug("No COMSOL reference data found at %s", csv_path)
        return None

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return {
        "time": data[:, 0],
        "potential": data[:, 1],
        "current": data[:, 2],
    }
