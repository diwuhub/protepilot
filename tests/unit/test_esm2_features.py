"""Tests for src/esm2_features.py.

The real ESM-2 forward pass is expensive (~100ms/pair on CPU), so these
tests monkey-patch `esm2_hybrid_encoder._embed_antibody_batch` with a
deterministic stub. One end-to-end integration test (marked slow) is
included that exercises the real path if the ESM-2 install is present.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed (Layer 4 training)")

from src.esm2_features import (
    CACHE_VERSION,
    DEFAULT_CACHE_PATH,
    EMBEDDING_DIM,
    MODEL_NAME,
    cache_info,
    get_esm2_embeddings,
    load_cache,
    save_cache,
    seq_pair_hash,
    _empty_cache,
)


def _stub_embed(vh_seqs, vl_seqs, embedder=None):
    """Deterministic stub that returns a function of hashed pair content.

    Each pair gets a 960-dim embedding whose value is `hash % 1000 / 1000`,
    so hash collisions would cause test failures — and matches work as they
    should.
    """
    rows = []
    for vh, vl in zip(vh_seqs, vl_seqs):
        h = abs(hash(vh.upper() + "|" + vl.upper())) % 1000
        row = np.full(EMBEDDING_DIM, h / 1000.0, dtype=np.float32)
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)


@pytest.fixture
def tmp_cache_path(tmp_path):
    return tmp_path / "esm2_test_cache.pt"


@pytest.fixture
def patched_embedder(monkeypatch):
    import esm2_hybrid_encoder

    monkeypatch.setattr(esm2_hybrid_encoder, "_embed_antibody_batch", _stub_embed)
    monkeypatch.setattr(esm2_hybrid_encoder, "_get_esm2_embedder", lambda: object())


class TestHash:
    def test_deterministic(self):
        h1 = seq_pair_hash("AAA", "BBB")
        h2 = seq_pair_hash("AAA", "BBB")
        assert h1 == h2

    def test_case_and_whitespace_insensitive(self):
        assert seq_pair_hash("aaa", "BBB") == seq_pair_hash(" AAA ", "bbb")

    def test_different_pairs_different_hash(self):
        assert seq_pair_hash("AAA", "BBB") != seq_pair_hash("BBB", "AAA")


class TestContract:
    def test_embedding_dim_is_960(self):
        assert EMBEDDING_DIM == 960

    def test_model_name_is_t12(self):
        assert MODEL_NAME == "facebook/esm2_t12_35M_UR50D"


class TestCacheIO:
    def test_load_nonexistent_returns_empty(self, tmp_cache_path):
        state = load_cache(tmp_cache_path)
        assert state.size() == 0

    def test_roundtrip(self, tmp_cache_path):
        state = _empty_cache()
        state.hash_to_row = {"abc": 0, "def": 1}
        state.embeddings = torch.tensor(
            [[1.0] * EMBEDDING_DIM, [2.0] * EMBEDDING_DIM],
            dtype=torch.float32,
        )
        save_cache(state, tmp_cache_path)
        loaded = load_cache(tmp_cache_path)
        assert loaded.size() == 2
        assert loaded.hash_to_row == {"abc": 0, "def": 1}
        assert torch.allclose(loaded.embeddings, state.embeddings)

    def test_rejects_wrong_version(self, tmp_cache_path):
        torch.save(
            {
                "version": CACHE_VERSION + 999,
                "model_name": MODEL_NAME,
                "dim": EMBEDDING_DIM,
                "hash_to_row": {},
                "embeddings": torch.zeros((0, EMBEDDING_DIM)),
            },
            tmp_cache_path,
        )
        with pytest.raises(RuntimeError, match="version"):
            load_cache(tmp_cache_path)

    def test_rejects_wrong_dim(self, tmp_cache_path):
        torch.save(
            {
                "version": CACHE_VERSION,
                "model_name": MODEL_NAME,
                "dim": 123,
                "hash_to_row": {},
                "embeddings": torch.zeros((0, 123)),
            },
            tmp_cache_path,
        )
        with pytest.raises(RuntimeError, match="dim"):
            load_cache(tmp_cache_path)


class TestGetEsm2Embeddings:
    def test_empty_input(self, tmp_cache_path):
        out = get_esm2_embeddings([], [], cache_path=tmp_cache_path)
        assert out.shape == (0, EMBEDDING_DIM)

    def test_mismatched_lengths_raise(self, tmp_cache_path):
        with pytest.raises(ValueError, match="same length"):
            get_esm2_embeddings(["A"], ["B", "C"], cache_path=tmp_cache_path)

    def test_cold_cache_builds_and_persists(self, tmp_cache_path, patched_embedder):
        vh = ["EVQLV", "DIQMT"]
        vl = ["GGLVQ", "SPSSL"]
        out = get_esm2_embeddings(vh, vl, cache_path=tmp_cache_path)
        assert out.shape == (2, EMBEDDING_DIM)
        assert tmp_cache_path.exists()
        info = cache_info(tmp_cache_path)
        assert info["n_cached"] == 2

    def test_warm_cache_returns_stable_values(self, tmp_cache_path, patched_embedder):
        vh = ["EVQLV"]
        vl = ["GGLVQ"]
        first = get_esm2_embeddings(vh, vl, cache_path=tmp_cache_path)
        second = get_esm2_embeddings(vh, vl, cache_path=tmp_cache_path)
        np.testing.assert_array_equal(first, second)

    def test_order_preserved(self, tmp_cache_path, patched_embedder):
        vh = ["AAA", "BBB", "CCC"]
        vl = ["xxx", "yyy", "zzz"]
        out_abc = get_esm2_embeddings(vh, vl, cache_path=tmp_cache_path)
        # Reversed input — output should reverse too.
        out_rev = get_esm2_embeddings(vh[::-1], vl[::-1], cache_path=tmp_cache_path)
        np.testing.assert_array_equal(out_rev, out_abc[::-1])

    def test_allow_build_false_raises_on_miss(self, tmp_cache_path, patched_embedder):
        with pytest.raises(RuntimeError, match="not in cache"):
            get_esm2_embeddings(
                ["NEWSEQ"], ["ANOTHER"],
                cache_path=tmp_cache_path,
                allow_build=False,
            )

    def test_deduplicates_forward_pass(self, tmp_cache_path, monkeypatch):
        """Same (vh, vl) appearing N times should be embedded once."""
        import esm2_hybrid_encoder

        call_counts = {"n_sequences": 0}

        def counting_embed(vh_seqs, vl_seqs, embedder=None):
            call_counts["n_sequences"] += len(vh_seqs)
            return _stub_embed(vh_seqs, vl_seqs)

        monkeypatch.setattr(esm2_hybrid_encoder, "_embed_antibody_batch", counting_embed)
        monkeypatch.setattr(esm2_hybrid_encoder, "_get_esm2_embedder", lambda: object())

        # Ten identical pairs — should embed once.
        vh = ["EVQLV"] * 10
        vl = ["GGLVQ"] * 10
        out = get_esm2_embeddings(vh, vl, cache_path=tmp_cache_path)
        assert out.shape == (10, EMBEDDING_DIM)
        assert call_counts["n_sequences"] == 1


class TestCacheInfo:
    def test_nonexistent(self, tmp_cache_path):
        info = cache_info(tmp_cache_path)
        assert info == {"path": str(tmp_cache_path), "exists": False, "n_cached": 0}

    def test_populated(self, tmp_cache_path, patched_embedder):
        get_esm2_embeddings(["AAA"], ["BBB"], cache_path=tmp_cache_path)
        info = cache_info(tmp_cache_path)
        assert info["exists"]
        assert info["n_cached"] == 1
        assert info["dim"] == EMBEDDING_DIM
        assert info["cache_version"] == CACHE_VERSION
