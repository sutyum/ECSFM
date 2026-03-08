# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from ecsfm.sim.reactions import (
    ReactionMechanism,
    ReactionStep,
    make_catalytic_mechanism,
    make_ec_mechanism,
    make_ece_mechanism,
    make_ee_mechanism,
    rde_velocity_profile,
)


class TestReactionStep:
    def test_e_step_valid(self):
        step = ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=0.0, k0=0.01)
        assert step.type == "E"

    def test_c_step_valid(self):
        step = ReactionStep(type="C", reactant_idx=1, product_idx=2, k_forward=1.0)
        assert step.type == "C"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="type must be"):
            ReactionStep(type="X", reactant_idx=0, product_idx=1)

    def test_invalid_k0(self):
        with pytest.raises(ValueError, match="k0 must be positive"):
            ReactionStep(type="E", reactant_idx=0, product_idx=1, k0=-1.0)


class TestReactionMechanism:
    def test_surface_rates_equilibrium(self):
        """At E=E0 with equal concentrations, net flux should be ~0."""
        mech = ReactionMechanism(
            steps=[ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=0.0, k0=0.01)],
            n_species=2,
        )
        C_surf = jnp.array([1.0, 1.0])
        rates = mech.compute_surface_rates(jnp.array(0.0), C_surf)
        assert jnp.allclose(rates, 0.0, atol=1e-10)

    def test_surface_rates_conservation(self):
        """Sum of surface rates should be zero (mass conservation)."""
        mech = ReactionMechanism(
            steps=[ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=0.0, k0=0.01)],
            n_species=2,
        )
        C_surf = jnp.array([1.0, 0.5])
        rates = mech.compute_surface_rates(jnp.array(0.3), C_surf)
        assert jnp.isclose(jnp.sum(rates), 0.0, atol=1e-12)

    def test_homogeneous_rates_forward(self):
        """C-type step: kf*C_R - kb*C_P."""
        mech = ReactionMechanism(
            steps=[
                ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=0.0, k0=0.01),
                ReactionStep(type="C", reactant_idx=1, product_idx=2, k_forward=1.0, k_backward=0.0),
            ],
            n_species=3,
        )
        C = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])  # (3, 3)
        rates = mech.compute_homogeneous_rates(C)
        # Species 1 consumed (rate = -kf*C1), species 2 produced (rate = +kf*C1)
        assert jnp.all(rates[1] < 0)
        assert jnp.all(rates[2] > 0)
        assert jnp.allclose(rates[0], 0.0)

    def test_homogeneous_equilibrium(self):
        """At equilibrium kf*C_R = kb*C_P, rates should be zero."""
        mech = ReactionMechanism(
            steps=[
                ReactionStep(type="E", reactant_idx=0, product_idx=1, E0=0.0, k0=0.01),
                ReactionStep(type="C", reactant_idx=1, product_idx=2, k_forward=2.0, k_backward=2.0),
            ],
            n_species=3,
        )
        C = jnp.array([[0.0], [1.0], [1.0]])  # C_R = C_P, kf = kb
        rates = mech.compute_homogeneous_rates(C)
        assert jnp.allclose(rates, 0.0, atol=1e-12)


class TestMechanismFactories:
    def test_ec_mechanism(self):
        mech = make_ec_mechanism(E0=0.0, k_forward=1.0)
        assert mech.n_species == 3
        assert len(mech.steps) == 2
        assert mech.steps[0].type == "E"
        assert mech.steps[1].type == "C"

    def test_ece_mechanism(self):
        mech = make_ece_mechanism()
        assert mech.n_species == 4
        assert len(mech.steps) == 3

    def test_catalytic_mechanism(self):
        mech = make_catalytic_mechanism(k_cat=5.0)
        assert mech.n_species == 2
        # C step regenerates species 0 from species 1
        assert mech.steps[1].reactant_idx == 1
        assert mech.steps[1].product_idx == 0

    def test_ee_mechanism(self):
        mech = make_ee_mechanism()
        assert mech.n_species == 3
        assert all(s.type == "E" for s in mech.steps)


class TestRDEVelocity:
    def test_velocity_at_surface(self):
        """Velocity at x=0 should be zero (no-slip condition)."""
        x = jnp.linspace(0.0, 0.05, 100)
        v = rde_velocity_profile(x, omega_rad_s=100.0)
        assert jnp.isclose(v[0], 0.0)

    def test_velocity_negative(self):
        """Velocity should be negative (toward electrode)."""
        x = jnp.linspace(0.001, 0.05, 100)
        v = rde_velocity_profile(x, omega_rad_s=100.0)
        assert jnp.all(v < 0)

    def test_velocity_quadratic(self):
        """Velocity should scale as x^2."""
        x = jnp.array([0.01, 0.02])
        v = rde_velocity_profile(x, omega_rad_s=100.0)
        # v(2x) / v(x) should be 4
        ratio = v[1] / v[0]
        assert jnp.isclose(ratio, 4.0, atol=1e-10)
