"""Reaction mechanism support: EC, ECE, EC' (catalytic), EE."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ReactionStep:
    """A single step in a reaction mechanism.

    type: "E" (electrochemical) or "C" (chemical/homogeneous)

    E-type fields: E0, k0, alpha, n_electrons
    C-type fields: k_forward, k_backward
    """

    type: str  # "E" or "C"
    reactant_idx: int
    product_idx: int
    # E-type parameters
    E0: float = 0.0
    k0: float = 0.01
    alpha: float = 0.5
    n_electrons: int = 1
    # C-type parameters
    k_forward: float = 0.0
    k_backward: float = 0.0

    def __post_init__(self):
        if self.type not in ("E", "C"):
            raise ValueError(f"type must be 'E' or 'C', got '{self.type}'")
        if self.reactant_idx < 0:
            raise ValueError(f"reactant_idx must be >= 0, got {self.reactant_idx}")
        if self.product_idx < 0:
            raise ValueError(f"product_idx must be >= 0, got {self.product_idx}")
        if self.type == "E":
            if self.k0 <= 0:
                raise ValueError(f"k0 must be positive, got {self.k0}")
            if not (0 <= self.alpha <= 1):
                raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
            if self.n_electrons < 1:
                raise ValueError(f"n_electrons must be >= 1, got {self.n_electrons}")


class ReactionMechanism(eqx.Module):
    """Multi-step reaction mechanism (EC, ECE, EC', EE).

    Each step is either electrochemical (E) or chemical (C).
    Species are indexed 0..n_species-1.
    """

    steps: tuple[ReactionStep, ...]
    n_species: int
    _f_const: float  # F/(RT)

    def __init__(
        self,
        steps: list[ReactionStep] | tuple[ReactionStep, ...],
        n_species: int,
        T: float = 298.15,
    ):
        if n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        steps_tuple = tuple(steps)
        if len(steps_tuple) == 0:
            raise ValueError("At least one reaction step is required")
        for s in steps_tuple:
            if s.reactant_idx >= n_species or s.product_idx >= n_species:
                raise ValueError(
                    f"Species index out of range: reactant={s.reactant_idx}, "
                    f"product={s.product_idx}, n_species={n_species}"
                )
        self.steps = steps_tuple
        self.n_species = n_species
        F = 96485.3321
        R = 8.314462618
        self._f_const = F / (R * T)

    def compute_surface_rates(
        self, E: jax.Array, C_surface: jax.Array
    ) -> jax.Array:
        """Production/consumption rates at electrode surface for all species.

        Only E-type steps contribute. Returns flux array of shape (n_species,).
        Positive = production, negative = consumption.

        Args:
            E: Applied potential (scalar)
            C_surface: Surface concentrations, shape (n_species,)

        Returns:
            rates: Net molar flux for each species, shape (n_species,)
        """
        dtype = C_surface.dtype
        rates = jnp.zeros(self.n_species, dtype=dtype)
        E = jnp.asarray(E, dtype=dtype)
        f = jnp.asarray(self._f_const, dtype=dtype)

        for step in self.steps:
            if step.type != "E":
                continue

            eta = E - jnp.asarray(step.E0, dtype=dtype)
            alpha = jnp.asarray(step.alpha, dtype=dtype)
            k0 = jnp.asarray(step.k0, dtype=dtype)
            n_e = jnp.asarray(float(step.n_electrons), dtype=dtype)

            k_ox = k0 * jnp.exp((1.0 - alpha) * n_e * f * eta)
            k_red = k0 * jnp.exp(-alpha * n_e * f * eta)

            # flux = k_ox * C_red - k_red * C_ox (convention: oxidation positive)
            flux = k_ox * C_surface[step.reactant_idx] - k_red * C_surface[step.product_idx]

            # Reactant (reduced form) is consumed, product (oxidized form) is produced
            rates = rates.at[step.reactant_idx].add(-flux)
            rates = rates.at[step.product_idx].add(flux)

        return rates

    def compute_homogeneous_rates(self, C: jax.Array) -> jax.Array:
        """Bulk chemical reaction rates for all species at all grid points.

        Only C-type steps contribute.

        Args:
            C: Concentrations, shape (n_species, nx)

        Returns:
            rates: Production rates, shape (n_species, nx)
        """
        dtype = C.dtype
        rates = jnp.zeros_like(C)

        for step in self.steps:
            if step.type != "C":
                continue

            kf = jnp.asarray(step.k_forward, dtype=dtype)
            kb = jnp.asarray(step.k_backward, dtype=dtype)

            # R -> P with rate kf*C_R - kb*C_P
            rate = kf * C[step.reactant_idx] - kb * C[step.product_idx]

            rates = rates.at[step.reactant_idx].add(-rate)
            rates = rates.at[step.product_idx].add(rate)

        return rates


def make_ec_mechanism(
    E0: float = 0.0,
    k0: float = 0.01,
    alpha: float = 0.5,
    k_forward: float = 1.0,
    k_backward: float = 0.0,
) -> ReactionMechanism:
    """Create an EC mechanism: A -e-> B -> C.

    Species 0=A (reactant), 1=B (intermediate), 2=C (product).
    """
    return ReactionMechanism(
        steps=[
            ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=E0, k0=k0, alpha=alpha),
            ReactionStep(type="C", reactant_idx=1, product_idx=2, k_forward=k_forward, k_backward=k_backward),
        ],
        n_species=3,
    )


def make_ece_mechanism(
    E0_1: float = 0.0,
    E0_2: float = -0.3,
    k0: float = 0.01,
    alpha: float = 0.5,
    k_forward: float = 1.0,
    k_backward: float = 0.0,
) -> ReactionMechanism:
    """Create an ECE mechanism: A -e-> B -> C -e-> D.

    Species 0=A, 1=B, 2=C, 3=D.
    """
    return ReactionMechanism(
        steps=[
            ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=E0_1, k0=k0, alpha=alpha),
            ReactionStep(type="C", reactant_idx=1, product_idx=2, k_forward=k_forward, k_backward=k_backward),
            ReactionStep(type="E", reactant_idx=2, product_idx=3, E0=E0_2, k0=k0, alpha=alpha),
        ],
        n_species=4,
    )


def make_catalytic_mechanism(
    E0: float = 0.0,
    k0: float = 0.01,
    alpha: float = 0.5,
    k_cat: float = 1.0,
) -> ReactionMechanism:
    """Create an EC' (catalytic) mechanism: A -e-> B, B -> A (catalytic regeneration).

    Species 0=A, 1=B. The C step regenerates A from B.
    """
    return ReactionMechanism(
        steps=[
            ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=E0, k0=k0, alpha=alpha),
            ReactionStep(type="C", reactant_idx=1, product_idx=0, k_forward=k_cat, k_backward=0.0),
        ],
        n_species=2,
    )


def make_ee_mechanism(
    E0_1: float = 0.0,
    E0_2: float = -0.3,
    k0_1: float = 0.01,
    k0_2: float = 0.01,
    alpha: float = 0.5,
) -> ReactionMechanism:
    """Create an EE mechanism: A -e-> B -e-> C.

    Species 0=A, 1=B, 2=C.
    """
    return ReactionMechanism(
        steps=[
            ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=E0_1, k0=k0_1, alpha=alpha),
            ReactionStep(type="E", reactant_idx=1, product_idx=2, E0=E0_2, k0=k0_2, alpha=alpha),
        ],
        n_species=3,
    )


def rde_velocity_profile(
    x: jax.Array, omega_rad_s: float, nu_cm2s: float = 0.01
) -> jax.Array:
    """Levich velocity profile for a rotating disk electrode.

    v(x) = -0.51 * omega^(3/2) * nu^(-1/2) * x^2

    Args:
        x: Distance from electrode surface (cm), shape (nx,)
        omega_rad_s: Angular velocity (rad/s)
        nu_cm2s: Kinematic viscosity (cm^2/s), default 0.01 for water

    Returns:
        v: Velocity profile (cm/s), shape (nx,). Negative = toward electrode.
    """
    dtype = x.dtype
    omega = jnp.asarray(omega_rad_s, dtype=dtype)
    nu = jnp.asarray(nu_cm2s, dtype=dtype)
    coeff = jnp.asarray(-0.51, dtype=dtype)
    return coeff * omega ** 1.5 * nu ** (-0.5) * x**2
