"""Adaptive time-stepping for the ECSFM electrochemical simulator.

Uses an embedded Euler-Heun (order 1/2) Richardson error estimate:
  - Take a full step of size dt
  - Take two half steps of size dt/2
  - Error = |full_step - two_half_steps|
  - Accept/reject and resize dt accordingly

Implicit diffusion is unconditionally stable and is *not* error-controlled;
only the explicit surface-kinetics terms participate in the error estimate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive time-stepping."""

    dt_min: float = 1e-10
    dt_max: float = 1e-2
    atol: float = 1e-6
    rtol: float = 1e-3
    safety_factor: float = 0.9
    max_growth: float = 2.0
    min_shrink: float = 0.5
