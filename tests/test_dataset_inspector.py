import numpy as np

from ecsfm.data.inspect import (
    load_sample_records,
    resolve_chunk_files,
    scan_dataset,
    select_random_global_indices,
    summarize_dataset,
)


def _make_params(
    *,
    max_species: int = 2,
    c_ox: tuple[float, float] = (1.0, 0.0),
    c_red: tuple[float, float] = (0.0, 0.0),
    alpha: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    p = np.zeros((7 * max_species,), dtype=np.float64)
    p[0:max_species] = np.log(np.asarray([1e-5, 1e-5]))
    p[max_species : 2 * max_species] = np.log(np.asarray([1e-5, 1e-5]))
    p[2 * max_species : 3 * max_species] = np.asarray(c_ox)
    p[3 * max_species : 4 * max_species] = np.asarray(c_red)
    p[4 * max_species : 5 * max_species] = np.asarray([0.0, 0.0])
    p[5 * max_species : 6 * max_species] = np.log(np.asarray([0.01, 0.01]))
    p[6 * max_species : 7 * max_species] = np.asarray(alpha)
    return p


def _write_chunk(
    path,
    *,
    ox: np.ndarray,
    red: np.ndarray,
    i: np.ndarray,
    e: np.ndarray,
    p: np.ndarray,
    task_id: np.ndarray | None = None,
    stage_id: np.ndarray | None = None,
    aug_id: np.ndarray | None = None,
):
    payload = {
        "ox": np.asarray(ox),
        "red": np.asarray(red),
        "i": np.asarray(i),
        "e": np.asarray(e),
        "p": np.asarray(p),
        "task_names": np.asarray(["t0", "t1", "t2"]),
        "stage_names": np.asarray(["s0", "s1", "s2"]),
        "augmentation_names": np.asarray(["a0", "a1", "a2"]),
    }
    if task_id is not None:
        payload["task_id"] = np.asarray(task_id)
    if stage_id is not None:
        payload["stage_id"] = np.asarray(stage_id)
    if aug_id is not None:
        payload["aug_id"] = np.asarray(aug_id)
    np.savez(path, **payload)


def test_resolve_chunk_files_numeric_order(tmp_path):
    rows = 1
    ox = np.zeros((rows, 6))
    red = np.zeros((rows, 6))
    i = np.zeros((rows, 5))
    e = np.zeros((rows, 5))
    p = np.zeros((rows, 14))

    _write_chunk(tmp_path / "chunk_10.npz", ox=ox, red=red, i=i, e=e, p=p)
    _write_chunk(tmp_path / "chunk_2.npz", ox=ox, red=red, i=i, e=e, p=p)

    chunks = resolve_chunk_files(tmp_path)
    assert [item.name for item in chunks] == ["chunk_2.npz", "chunk_10.npz"]


def test_summary_tracks_distribution_and_sanity_flags(tmp_path):
    ox0 = np.asarray(
        [
            [0.2, 0.2, 0.2, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        ]
    )
    red0 = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
        ]
    )
    i0 = np.asarray(
        [
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    e0 = np.asarray(
        [
            [0.2, 0.4, 0.6, 0.4, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    p0 = np.asarray(
        [
            _make_params(),
            _make_params(c_ox=(-0.1, 0.0), alpha=(1.2, 0.5)),
        ]
    )
    _write_chunk(
        tmp_path / "chunk_0.npz",
        ox=ox0,
        red=red0,
        i=i0,
        e=e0,
        p=p0,
        task_id=np.asarray([0, 1]),
        stage_id=np.asarray([0, 1]),
        aug_id=np.asarray([0, 2]),
    )

    ox1 = np.asarray([[-0.2, 0.2, 0.2, 0.0, 0.0, 0.0]])
    red1 = np.asarray([[0.0, 0.0, 0.0, 0.2, 0.2, 0.2]])
    i1 = np.asarray([[np.nan, 0.5, 0.0, -0.5, 0.0]])
    e1 = np.asarray([[0.1, 0.2, 0.3, 0.2, 0.1]])
    p1 = np.asarray([_make_params()])
    _write_chunk(
        tmp_path / "chunk_1.npz",
        ox=ox1,
        red=red1,
        i=i1,
        e=e1,
        p=p1,
        task_id=np.asarray([2]),
        stage_id=np.asarray([2]),
        aug_id=np.asarray([1]),
    )

    chunks, layout, labels = scan_dataset(tmp_path)
    summary = summarize_dataset(chunks, layout, labels)

    assert summary["total_rows"] == 3
    assert summary["layout"]["max_species"] == 2
    assert summary["layout"]["nx"] == 3
    assert summary["diagnostics"]["nonfinite_rows"] == 1
    assert summary["diagnostics"]["negative_profile_rows"] == 1
    assert summary["diagnostics"]["negative_bulk_rows"] == 1
    assert summary["diagnostics"]["invalid_alpha_rows"] == 1
    assert summary["diagnostics"]["flat_current_rows"] == 1
    assert summary["diagnostics"]["flat_potential_rows"] == 1

    task_dist = {item["id"]: item["count"] for item in summary["distribution"]["task"]}
    stage_dist = {item["id"]: item["count"] for item in summary["distribution"]["stage"]}
    aug_dist = {item["id"]: item["count"] for item in summary["distribution"]["augmentation"]}
    assert task_dist[0] == 1
    assert task_dist[1] == 1
    assert task_dist[2] == 1
    assert stage_dist[0] == 1
    assert stage_dist[1] == 1
    assert stage_dist[2] == 1
    assert aug_dist[0] == 1
    assert aug_dist[1] == 1
    assert aug_dist[2] == 1


def test_select_random_global_indices_replacement_behavior():
    picks = select_random_global_indices(total_rows=10, n_samples=5, seed=7)
    assert len(picks) == 5
    assert len(set(picks)) == 5

    picks = select_random_global_indices(total_rows=3, n_samples=8, seed=7)
    assert len(picks) == 8
    assert all(0 <= item < 3 for item in picks)


def test_load_sample_records_uses_global_indices(tmp_path):
    ox = np.arange(18, dtype=np.float64).reshape(3, 6)
    red = ox + 0.5
    i = np.arange(15, dtype=np.float64).reshape(3, 5)
    e = i * 0.01
    p = np.vstack([_make_params(), _make_params(), _make_params()])

    _write_chunk(
        tmp_path / "chunk_0.npz",
        ox=ox[:2],
        red=red[:2],
        i=i[:2],
        e=e[:2],
        p=p[:2],
        task_id=np.asarray([0, 1]),
        stage_id=np.asarray([0, 1]),
        aug_id=np.asarray([0, 1]),
    )
    _write_chunk(
        tmp_path / "chunk_1.npz",
        ox=ox[2:],
        red=red[2:],
        i=i[2:],
        e=e[2:],
        p=p[2:],
        task_id=np.asarray([2]),
        stage_id=np.asarray([2]),
        aug_id=np.asarray([2]),
    )

    chunks, _, _ = scan_dataset(tmp_path)
    records = load_sample_records(chunks, indices=[2, 0])

    assert [rec.global_index for rec in records] == [2, 0]
    assert [rec.chunk_path.name for rec in records] == ["chunk_1.npz", "chunk_0.npz"]
    assert [rec.chunk_row for rec in records] == [0, 0]
    assert records[0].task_id == 2
    assert records[1].task_id == 0
