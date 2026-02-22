import numpy as np

from ecsfm.fm.posterior import (
    CEMPosteriorConfig,
    compute_reliability_metrics,
    run_cem_posterior,
)


def test_run_cem_posterior_recovers_quadratic_optimum():
    target = np.array([1.2, -0.7, 0.35, 2.0], dtype=np.float32)

    def loglik(samples: np.ndarray) -> np.ndarray:
        return -np.sum((samples - target[None, :]) ** 2, axis=1)

    result = run_cem_posterior(
        log_likelihood_fn=loglik,
        dim=target.shape[0],
        config=CEMPosteriorConfig(
            n_particles=96,
            n_iterations=6,
            elite_fraction=0.25,
        ),
        seed=7,
    )

    mean = np.asarray(result["posterior_mean"])
    assert mean.shape == target.shape
    np.testing.assert_allclose(mean, target, atol=0.2, rtol=0.0)


def test_run_cem_posterior_respects_known_dimensions():
    target = np.array([0.5, -1.5, 2.2], dtype=np.float32)
    known_mask = np.array([False, True, False], dtype=bool)
    known_values = np.array([0.0, -1.5, 0.0], dtype=np.float32)

    def loglik(samples: np.ndarray) -> np.ndarray:
        return -np.sum((samples - target[None, :]) ** 2, axis=1)

    result = run_cem_posterior(
        log_likelihood_fn=loglik,
        dim=target.shape[0],
        known_values=known_values,
        known_mask=known_mask,
        config=CEMPosteriorConfig(
            n_particles=80,
            n_iterations=5,
            elite_fraction=0.3,
        ),
        seed=11,
    )

    samples = np.asarray(result["samples"])
    assert np.allclose(samples[:, 1], -1.5)
    assert np.isclose(np.asarray(result["posterior_mean"])[1], -1.5)


def test_reliability_metrics_are_well_calibrated_for_gaussian_noise():
    rng = np.random.default_rng(0)
    n = 5000
    sigma = 0.2
    truth = np.sin(np.linspace(0.0, 6.0 * np.pi, n))
    obs = truth + rng.normal(0.0, sigma, size=n)
    pred_mean = truth
    pred_std = np.full((n,), sigma)

    metrics = compute_reliability_metrics(obs, pred_mean, pred_std)

    assert 0.62 <= metrics["coverage_1sigma"] <= 0.74
    assert 0.92 <= metrics["coverage_2sigma"] <= 0.98
    assert np.isfinite(metrics["nll"])
    assert 0.0 <= metrics["reliability_score"] <= 100.0
