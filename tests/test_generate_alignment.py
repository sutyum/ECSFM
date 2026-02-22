import jax
import numpy as np

from ecsfm.data import generate


def test_dataset_outputs_stay_aligned_with_params(monkeypatch):
    def fake_run_single_sim(args):
        _, _, params, nx = args
        D_ox, *_ = params
        marker = float(D_ox[0])
        max_species = len(D_ox)
        c_ox = np.full((max_species, nx), marker, dtype=float)
        c_red = np.full((max_species, nx), -marker, dtype=float)
        i_hist = np.full(200, marker, dtype=float)
        signal = np.full(200, marker, dtype=float)
        return c_ox, c_red, i_hist, signal

    monkeypatch.setattr(generate, "_run_single_sim", fake_run_single_sim)

    c_ox, c_red, curr, sigs, params, task_id, stage_id, aug_id = generate.generate_multi_species_dataset(
        n_samples=8,
        key=jax.random.PRNGKey(0),
        max_species=3,
        nx=10,
        max_workers=1,
        backend="process_pool",
        include_invariant_pairs=False,
    )

    c_ox = np.asarray(c_ox)
    c_red = np.asarray(c_red)
    curr = np.asarray(curr)
    sigs = np.asarray(sigs)
    params = np.asarray(params)
    task_id = np.asarray(task_id)
    stage_id = np.asarray(stage_id)
    aug_id = np.asarray(aug_id)

    assert c_ox.shape == (8, 30)
    assert c_red.shape == (8, 30)
    assert curr.shape == (8, 200)
    assert sigs.shape == (8, 200)
    assert task_id.shape == (8,)
    assert stage_id.shape == (8,)
    assert aug_id.shape == (8,)
    assert np.all(aug_id == 0)

    markers_from_outputs = c_ox[:, 0]
    markers_from_params = np.exp(params[:, 0])
    assert np.allclose(markers_from_outputs, markers_from_params)


def test_gpu_batch_backend_smoke_shapes():
    c_ox, c_red, curr, sigs, params, task_id, stage_id, aug_id = generate.generate_multi_species_dataset(
        n_samples=4,
        key=jax.random.PRNGKey(11),
        max_species=2,
        nx=8,
        backend="gpu_batch",
        sim_steps=32,
        device_batch_size=2,
        include_invariant_pairs=False,
        target_sig_len=32,
    )

    c_ox = np.asarray(c_ox)
    c_red = np.asarray(c_red)
    curr = np.asarray(curr)
    sigs = np.asarray(sigs)
    params = np.asarray(params)
    task_id = np.asarray(task_id)
    stage_id = np.asarray(stage_id)
    aug_id = np.asarray(aug_id)

    assert c_ox.shape == (4, 16)
    assert c_red.shape == (4, 16)
    assert curr.shape == (4, 32)
    assert sigs.shape == (4, 32)
    assert params.shape == (4, 14)
    assert task_id.shape == (4,)
    assert stage_id.shape == (4,)
    assert aug_id.shape == (4,)
    assert np.all(np.isfinite(c_ox))
    assert np.all(np.isfinite(c_red))
    assert np.all(np.isfinite(curr))
    assert np.all(np.isfinite(sigs))
