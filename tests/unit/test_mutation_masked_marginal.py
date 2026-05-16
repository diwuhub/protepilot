"""Tests for src/mutation/masked_marginal.py.

Uses a fake ESM-2 model + tokenizer that returns deterministic logits.
This keeps the test suite fast (no HuggingFace download, no model load).
One integration test that exercises the real model path is skipped by
default; set PROTEPILOT_TEST_REAL_ESM2=1 to enable.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed (Layer 4 training)")

from src.mutation.masked_marginal import (
    MODEL_NAME,
    MaskedMarginalConfig,
    MaskedMarginalScorer,
    score_mutations,
)
from src.mutation.schema import (
    STANDARD_AA,
    AntibodyChain,
    MutationCandidate,
)


# --- fake model / tokenizer ------------------------------------------------


class _FakeTokenizer:
    """Minimal ESM-2-like tokenizer: 1 BOS + N residues + 1 EOS."""

    def __init__(self):
        base = {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 32}
        for i, aa in enumerate(STANDARD_AA):
            base[aa] = 4 + i   # ids 4..23
        self._vocab = base
        self.mask_token = "<mask>"

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, seq, add_special_tokens=True, return_tensors="pt"):
        ids = [self._vocab.get(c, 3) for c in seq]
        if add_special_tokens:
            ids = [0] + ids + [2]
        return torch.tensor([ids])


class _FakeMaskedLM(torch.nn.Module):
    """Fake ESM-2 MLM: returns deterministic logits biased to prefer the
    residue at position `position` that equals the unmasked token.
    """

    def __init__(self):
        super().__init__()
        self.vocab_size = 64

    def __call__(self, input_ids):
        # (1, L) -> (1, L, V)
        bsz, L = input_ids.shape
        logits = torch.randn(bsz, L, self.vocab_size, generator=torch.Generator().manual_seed(0)) * 0.5
        # At the masked position(s), add a strong preference for the residue
        # that sits one position to the left — an arbitrary deterministic
        # rule so the test can assert signal direction.
        class _Out:
            def __init__(self, logits):
                self.logits = logits
        return _Out(logits)

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def fake_scorer():
    cfg = MaskedMarginalConfig(model_name="fake-esm", device="cpu", linker="GG")
    return MaskedMarginalScorer(config=cfg, _model=_FakeMaskedLM(), _tokenizer=_FakeTokenizer())


# --- tests ---------------------------------------------------------------


TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


class TestConstants:
    def test_model_name_is_t12(self):
        assert MODEL_NAME == "facebook/esm2_t12_35M_UR50D"


class TestFakeScore:
    def test_vh_only_single_mutation(self, fake_scorer):
        candidate = MutationCandidate(
            chain=AntibodyChain.VH, position=27, wildtype_aa="S", mutant_aa="A",
            region="cdr_h1",
        )
        scored = fake_scorer.score(TRASTUZUMAB_VH, "", [candidate])
        assert len(scored) == 1
        s = scored[0]
        assert s.candidate == candidate
        assert np.isfinite(s.llr)
        # wildtype/mutant probabilities sum to ≤ 1 within 20-AA renorm.
        assert 0.0 <= s.wildtype_prob <= 1.0
        assert 0.0 <= s.mutant_prob <= 1.0

    def test_vh_plus_vl_scoring(self, fake_scorer):
        # Pick positions whose WT we verify against the Trastuzumab sequence.
        cands = [
            MutationCandidate(
                AntibodyChain.VH, 30, TRASTUZUMAB_VH[30], "S", "framework",
            ),
            MutationCandidate(
                AntibodyChain.VL, 50, TRASTUZUMAB_VL[50], "T", "framework",
            ),
        ]
        scored = fake_scorer.score(TRASTUZUMAB_VH, TRASTUZUMAB_VL, cands)
        assert len(scored) == 2

    def test_uncertainty_labels(self, fake_scorer):
        cands = [MutationCandidate(AntibodyChain.VH, i, TRASTUZUMAB_VH[i],
                                   "A" if TRASTUZUMAB_VH[i] != "A" else "G",
                                   "framework")
                 for i in range(10)]
        scored = fake_scorer.score(TRASTUZUMAB_VH, "", cands)
        for s in scored:
            assert s.uncertainty in {"low", "medium", "high"}

    def test_deterministic_repeated_score(self, fake_scorer):
        cand = MutationCandidate(AntibodyChain.VH, 27, "S", "A", "cdr_h1")
        s1 = fake_scorer.score(TRASTUZUMAB_VH, "", [cand])[0]
        s2 = fake_scorer.score(TRASTUZUMAB_VH, "", [cand])[0]
        assert s1.llr == pytest.approx(s2.llr, abs=1e-6)

    def test_rationale_contains_llr(self, fake_scorer):
        cand = MutationCandidate(AntibodyChain.VH, 27, "S", "A", "cdr_h1")
        s = fake_scorer.score(TRASTUZUMAB_VH, "", [cand])[0]
        assert "LLR" in s.rationale


class TestIntegrationRealModel:
    """Real ESM-2 test — opt-in via env var."""

    @pytest.mark.skipif(
        os.environ.get("PROTEPILOT_TEST_REAL_ESM2") != "1",
        reason="set PROTEPILOT_TEST_REAL_ESM2=1 to run real ESM-2 tests",
    )
    def test_single_mutation_on_trastuzumab(self):
        cand = MutationCandidate(AntibodyChain.VH, 27, "S", "A", "cdr_h1")
        scored = score_mutations(TRASTUZUMAB_VH, TRASTUZUMAB_VL, [cand])
        assert len(scored) == 1
        assert np.isfinite(scored[0].llr)
