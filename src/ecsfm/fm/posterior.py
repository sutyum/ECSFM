from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ecsfm.fm.train import _build_param_input, _build_signal_input, integrate_flow


@dataclass(frozen=True)
class CEMPosteriorConfig:
    n_particles: int = 96
    n_iterations: int = 6
    elite_fraction: float = 0.25
    init_std: float = 1.0
    min_std: float = 0.05
    max_std: float = 2.5
    clip_norm: float = 4.0


@dataclass(frozen=True)
class PosteriorInferenceConfig:
    cem: CEMPosteriorConfig = CEMPosteriorConfig()
    n_mc_per_particle: int = 4
    n_integration_steps: int = 100
    obs_noise_std: float = 0.25


def _to_bool_mask(mask: np.ndarray | None, dim: int) -> np.ndarray:
    if mask is None:
        return np.zeros((dim,), dtype=bool)
    out = np.asarray(mask, dtype=bool).reshape(-1)
    if out.shape[0] != dim:
        raise ValueError(f"Mask length mismatch: expected {dim}, got {out.shape[0]}")
    return out


def run_cem_posterior(
    log_likelihood_fn: Callable[[np.ndarray], np.ndarray],
    dim: int,
    *,
    config: CEMPosteriorConfig | None = None,
    known_values: np.ndarray | None = None,
    known_mask: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Cross-entropy posterior approximation over a diagonal Gaussian proposal."""
    cfg = config or CEMPosteriorConfig()
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if cfg.n_particles < 8:
        raise ValueError(f"n_particles must be >= 8, got {cfg.n_particles}")
    if cfg.n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {cfg.n_iterations}")
    if not (0.0 < cfg.elite_fraction <= 1.0):
        raise ValueError(f"elite_fraction must be in (0,1], got {cfg.elite_fraction}")
    if cfg.min_std <= 0.0 or cfg.max_std <= 0.0:
        raise ValueError("min_std and max_std must be positive")
    if cfg.min_std > cfg.max_std:
        raise ValueError("min_std cannot exceed max_std")
    if cfg.init_std <= 0.0:
        raise ValueError(f"init_std must be positive, got {cfg.init_std}")
    if cfg.clip_norm <= 0.0:
        raise ValueError(f"clip_norm must be positive, got {cfg.clip_norm}")

    rng = np.random.default_rng(seed)

    known_mask_arr = _to_bool_mask(known_mask, dim)
    known_values_arr = np.zeros((dim,), dtype=np.float32)
    if known_values is not None:
        known_values_arr = np.asarray(known_values, dtype=np.float32).reshape(-1)
        if known_values_arr.shape[0] != dim:
            raise ValueError(f"known_values length mismatch: expected {dim}, got {known_values_arr.shape[0]}")

    mean = np.zeros((dim,), dtype=np.float32)
    std = np.full((dim,), float(cfg.init_std), dtype=np.float32)
    mean[known_mask_arr] = known_values_arr[known_mask_arr]
    std[known_mask_arr] = cfg.min_std

    n_elite = max(1, int(np.ceil(cfg.elite_fraction * cfg.n_particles)))

    for _ in range(cfg.n_iterations):
        samples = rng.normal(loc=mean, scale=std, size=(cfg.n_particles, dim)).astype(np.float32)
        samples = np.clip(samples, -cfg.clip_norm, cfg.clip_norm)
        samples[:, known_mask_arr] = known_values_arr[known_mask_arr]

        loglik = np.asarray(log_likelihood_fn(samples), dtype=np.float64).reshape(-1)
        if loglik.shape[0] != cfg.n_particles:
            raise ValueError(
                f"log_likelihood_fn must return shape ({cfg.n_particles},), got {loglik.shape}"
            )

        elite_idx = np.argsort(loglik)[-n_elite:]
        elites = samples[elite_idx]
        mean = np.mean(elites, axis=0).astype(np.float32)
        std = np.std(elites, axis=0).astype(np.float32)
        std = np.clip(std, cfg.min_std, cfg.max_std)
        mean[known_mask_arr] = known_values_arr[known_mask_arr]
        std[known_mask_arr] = cfg.min_std

    final_samples = rng.normal(loc=mean, scale=std, size=(cfg.n_particles, dim)).astype(np.float32)
    final_samples = np.clip(final_samples, -cfg.clip_norm, cfg.clip_norm)
    final_samples[:, known_mask_arr] = known_values_arr[known_mask_arr]
    final_loglik = np.asarray(log_likelihood_fn(final_samples), dtype=np.float64).reshape(-1)

    stable = final_loglik - float(np.max(final_loglik))
    weights = np.exp(stable)
    weights = weights / np.maximum(np.sum(weights), 1e-12)

    mean_post = np.sum(final_samples * weights[:, None], axis=0)
    second = np.sum((final_samples**2) * weights[:, None], axis=0)
    std_post = np.sqrt(np.maximum(second - mean_post**2, 1e-12))

    return {
        "samples": final_samples,
        "log_likelihood": final_loglik.astype(np.float32),
        "weights": weights.astype(np.float32),
        "proposal_mean": mean.astype(np.float32),
        "proposal_std": std.astype(np.float32),
        "posterior_mean": mean_post.astype(np.float32),
        "posterior_std": std_post.astype(np.float32),
    }


def compute_reliability_metrics(
    observed: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    *,
    obs_mask: np.ndarray | None = None,
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float).reshape(-1)
    pred_mean = np.asarray(pred_mean, dtype=float).reshape(-1)
    pred_std = np.asarray(pred_std, dtype=float).reshape(-1)
    if not (observed.shape[0] == pred_mean.shape[0] == pred_std.shape[0]):
        raise ValueError("observed, pred_mean, and pred_std must have matching lengths")

    if obs_mask is None:
        mask = np.ones_like(observed, dtype=bool)
    else:
        mask = np.asarray(obs_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != observed.shape[0]:
            raise ValueError("obs_mask length mismatch")

    if not np.any(mask):
        raise ValueError("obs_mask has no observed points")

    obs = observed[mask]
    mu = pred_mean[mask]
    sigma = np.maximum(pred_std[mask], 1e-6)
    resid = obs - mu

    rmse = float(np.sqrt(np.mean(resid**2)))
    obs_range = float(np.ptp(obs))
    nrmse = rmse / max(obs_range, 1e-8)
    nll = float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * (resid**2) / (sigma**2)))

    z_abs = np.abs(resid) / sigma
    cov1 = float(np.mean(z_abs <= 1.0))
    cov2 = float(np.mean(z_abs <= 2.0))
    cal_error = float(abs(cov1 - 0.6827) + abs(cov2 - 0.9545))
    sharpness = float(np.mean(sigma) / max(obs_range, 1e-8))

    score = 100.0 * (
        0.45 * float(np.exp(-nrmse))
        + 0.30 * float(np.exp(-nll))
        + 0.15 * float(np.exp(-4.0 * cal_error))
        + 0.10 * float(np.exp(-sharpness))
    )
    score = float(np.clip(score, 0.0, 100.0))

    return {
        "nrmse": nrmse,
        "nll": nll,
        "coverage_1sigma": cov1,
        "coverage_2sigma": cov2,
        "calibration_error": cal_error,
        "sharpness": sharpness,
        "reliability_score": score,
    }


def _resample_1d(trace: np.ndarray, target_len: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    if trace.shape[0] == target_len:
        return trace
    src = np.linspace(0.0, 1.0, trace.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(dst, src, trace).astype(np.float32)


def _evaluate_particle_batch(
    model,
    normalizers: tuple[jax.Array, ...],
    geometry: dict[str, int | bool],
    particles_core_norm: np.ndarray,
    e_norm_base: np.ndarray,
    obs_current: np.ndarray,
    obs_mask: np.ndarray,
    *,
    config: PosteriorInferenceConfig,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean, x_std, e_mean, e_std, p_mean, p_std = normalizers
    del e_mean, e_std, p_mean, p_std

    state_dim = int(geometry["state_dim"])
    max_species = int(geometry["max_species"])
    nx = int(geometry["nx"])
    target_len = int(geometry["target_len"])
    signal_channels = int(geometry["signal_channels"])
    param_mask_features = bool(geometry["param_mask_features"])

    particles_core_norm = np.asarray(particles_core_norm, dtype=np.float32)
    n_particles = particles_core_norm.shape[0]
    n_mc = int(config.n_mc_per_particle)
    if n_particles <= 0:
        raise ValueError("Need at least one particle")
    if n_mc <= 0:
        raise ValueError("n_mc_per_particle must be positive")

    p_norm = jnp.asarray(particles_core_norm, dtype=jnp.float32)
    p_input = _build_param_input(
        params_norm=p_norm,
        param_mask=jnp.ones_like(p_norm),
        append_mask_features=param_mask_features,
    )

    e_batch_base = jnp.repeat(jnp.asarray(e_norm_base[None, :], dtype=jnp.float32), repeats=n_particles, axis=0)
    e_input = _build_signal_input(
        signal_norm=e_batch_base,
        signal_mask=None,
        signal_channels=signal_channels,
    )

    p_input_mc = jnp.repeat(p_input, repeats=n_mc, axis=0)
    e_input_mc = jnp.repeat(e_input, repeats=n_mc, axis=0)
    key_x0 = key
    x0 = jax.random.normal(key_x0, shape=(n_particles * n_mc, state_dim), dtype=jnp.float32)
    x_gen = integrate_flow(
        model,
        x0=x0,
        E=e_input_mc,
        p=p_input_mc,
        n_steps=int(config.n_integration_steps),
    )
    x_denorm = x_gen * jnp.asarray(x_std, dtype=jnp.float32) + jnp.asarray(x_mean, dtype=jnp.float32)
    species_state_dim = max_species * nx
    currents = np.asarray(x_denorm[:, 2 * species_state_dim :], dtype=np.float32)
    currents = currents.reshape(n_particles, n_mc, target_len)

    pred_mean = np.mean(currents, axis=1)
    pred_std = np.std(currents, axis=1)

    obs = np.asarray(obs_current, dtype=np.float32).reshape(1, -1)
    mask = np.asarray(obs_mask, dtype=np.float32).reshape(1, -1)
    var = pred_std**2 + float(config.obs_noise_std) ** 2
    var = np.maximum(var, 1e-8)
    resid = obs - pred_mean
    ll_pointwise = -0.5 * (resid**2 / var + np.log(2.0 * np.pi * var))
    obs_count = max(float(np.sum(mask)), 1.0)
    loglik = np.sum(ll_pointwise * mask, axis=1) / obs_count
    return loglik.astype(np.float32), pred_mean.astype(np.float32), pred_std.astype(np.float32)


def infer_parameter_posterior(
    model,
    normalizers: tuple[jax.Array, ...],
    geometry: dict[str, int | bool],
    observed_current: np.ndarray,
    applied_signal: np.ndarray,
    *,
    known_p_core: np.ndarray | None = None,
    known_p_mask: np.ndarray | None = None,
    obs_mask: np.ndarray | None = None,
    config: PosteriorInferenceConfig | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray | dict[str, float]]:
    cfg = config or PosteriorInferenceConfig()
    x_mean, x_std, e_mean, e_std, p_mean, p_std = normalizers
    target_len = int(geometry["target_len"])
    phys_dim_core = int(geometry["phys_dim_core"])

    obs_current_rs = _resample_1d(observed_current, target_len)
    e_rs = _resample_1d(applied_signal, target_len)
    if obs_mask is None:
        obs_mask_rs = np.ones((target_len,), dtype=np.float32)
    else:
        obs_mask_rs = _resample_1d(obs_mask, target_len)
        obs_mask_rs = (obs_mask_rs >= 0.5).astype(np.float32)
        if float(np.sum(obs_mask_rs)) < 1.0:
            raise ValueError("obs_mask has no observed points after resampling")

    e_norm_base = np.asarray((jnp.asarray(e_rs) - e_mean) / e_std, dtype=np.float32)

    known_mask_arr = _to_bool_mask(known_p_mask, phys_dim_core)
    if known_p_core is None:
        known_norm_arr = np.zeros((phys_dim_core,), dtype=np.float32)
    else:
        known_raw_arr = np.asarray(known_p_core, dtype=np.float32).reshape(-1)
        if known_raw_arr.shape[0] != phys_dim_core:
            raise ValueError(
                f"known_p_core length mismatch: expected {phys_dim_core}, got {known_raw_arr.shape[0]}"
            )
        known_norm_arr = np.asarray((jnp.asarray(known_raw_arr) - p_mean) / p_std, dtype=np.float32)

    master_key = jax.random.PRNGKey(np.uint32(seed))

    def _loglik_fn(samples_norm: np.ndarray) -> np.ndarray:
        nonlocal master_key
        master_key, eval_key = jax.random.split(master_key)
        loglik, _, _ = _evaluate_particle_batch(
            model=model,
            normalizers=normalizers,
            geometry=geometry,
            particles_core_norm=samples_norm,
            e_norm_base=e_norm_base,
            obs_current=obs_current_rs,
            obs_mask=obs_mask_rs,
            config=cfg,
            key=eval_key,
        )
        return loglik

    cem_result = run_cem_posterior(
        log_likelihood_fn=_loglik_fn,
        dim=phys_dim_core,
        config=cfg.cem,
        known_values=known_norm_arr,
        known_mask=known_mask_arr,
        seed=seed,
    )

    posterior_samples_norm = np.asarray(cem_result["samples"], dtype=np.float32)
    master_key, final_key = jax.random.split(master_key)
    loglik, pred_mean_particles, pred_std_particles = _evaluate_particle_batch(
        model=model,
        normalizers=normalizers,
        geometry=geometry,
        particles_core_norm=posterior_samples_norm,
        e_norm_base=e_norm_base,
        obs_current=obs_current_rs,
        obs_mask=obs_mask_rs,
        config=cfg,
        key=final_key,
    )

    stable = loglik - float(np.max(loglik))
    weights = np.exp(stable)
    weights = weights / np.maximum(np.sum(weights), 1e-12)

    post_mean_norm = np.sum(posterior_samples_norm * weights[:, None], axis=0)
    post_second_norm = np.sum((posterior_samples_norm**2) * weights[:, None], axis=0)
    post_std_norm = np.sqrt(np.maximum(post_second_norm - post_mean_norm**2, 1e-12))

    post_mean_raw = np.asarray(jnp.asarray(post_mean_norm) * p_std + p_mean, dtype=np.float32)
    post_std_raw = np.asarray(jnp.asarray(post_std_norm) * p_std, dtype=np.float32)
    post_samples_raw = np.asarray(jnp.asarray(posterior_samples_norm) * p_std + p_mean, dtype=np.float32)

    pred_mean = np.sum(pred_mean_particles * weights[:, None], axis=0)
    pred_second = np.sum((pred_std_particles**2 + pred_mean_particles**2) * weights[:, None], axis=0)
    pred_std = np.sqrt(np.maximum(pred_second - pred_mean**2, 1e-12))

    reliability = compute_reliability_metrics(
        observed=obs_current_rs,
        pred_mean=pred_mean,
        pred_std=pred_std,
        obs_mask=obs_mask_rs,
    )

    return {
        "posterior_samples_norm": posterior_samples_norm,
        "posterior_samples_raw": post_samples_raw,
        "posterior_weights": weights.astype(np.float32),
        "posterior_mean_norm": post_mean_norm.astype(np.float32),
        "posterior_std_norm": post_std_norm.astype(np.float32),
        "posterior_mean_raw": post_mean_raw,
        "posterior_std_raw": post_std_raw,
        "predictive_mean_current": pred_mean.astype(np.float32),
        "predictive_std_current": pred_std.astype(np.float32),
        "observed_current": obs_current_rs.astype(np.float32),
        "observed_mask": obs_mask_rs.astype(np.float32),
        "applied_signal": e_rs.astype(np.float32),
        "reliability": reliability,
    }
