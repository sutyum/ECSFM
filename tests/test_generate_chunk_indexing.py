import os

import ecsfm.data.generate as generate
from ecsfm.data.generate import infer_next_chunk_index, reserve_chunk_lock


def test_infer_next_chunk_index_empty_dir(tmp_path):
    assert infer_next_chunk_index(str(tmp_path)) == 0


def test_infer_next_chunk_index_with_existing_chunks(tmp_path):
    (tmp_path / "chunk_0.npz").write_bytes(b"")
    (tmp_path / "chunk_2.npz").write_bytes(b"")
    (tmp_path / "chunk_10.npz").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    assert infer_next_chunk_index(str(tmp_path)) == 11


def test_infer_next_chunk_index_accounts_for_locks(tmp_path):
    (tmp_path / "chunk_3.npz").write_bytes(b"")
    (tmp_path / "chunk_4.lock").write_text("pid=123", encoding="utf-8")
    assert infer_next_chunk_index(str(tmp_path)) == 5


def test_reserve_chunk_lock_skips_taken_indices(tmp_path):
    (tmp_path / "chunk_0.npz").write_bytes(b"")
    (tmp_path / "chunk_1.lock").write_text("pid=999", encoding="utf-8")

    chunk_idx, lock_path = reserve_chunk_lock(str(tmp_path), start_idx=0)
    try:
        assert chunk_idx == 2
        assert lock_path.endswith("chunk_2.lock")
        assert (tmp_path / "chunk_2.lock").exists()
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def test_backend_auto_resolves_to_cpu_process_pool(monkeypatch):
    monkeypatch.setattr(generate.jax, "default_backend", lambda: "cpu")
    assert generate._resolve_generation_backend("auto") == "process_pool"


def test_backend_auto_resolves_to_gpu_batch(monkeypatch):
    monkeypatch.setattr(generate.jax, "default_backend", lambda: "gpu")
    assert generate._resolve_generation_backend("auto") == "gpu_batch"


def test_progress_mode_explicit():
    assert generate._resolve_progress_mode("off") is False
    assert generate._resolve_progress_mode("on") is True


def test_progress_mode_auto_respects_tty_and_env(monkeypatch):
    monkeypatch.setattr(generate.sys.stderr, "isatty", lambda: True)
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    assert generate._resolve_progress_mode("auto") is True

    monkeypatch.setenv("TQDM_DISABLE", "1")
    assert generate._resolve_progress_mode("auto") is False
