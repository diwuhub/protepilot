"""ESM-2 masked-marginal scoring (Meier et al., NeurIPS 2021).

The Meier method scores a single substitution by comparing the masked
probability of the mutant AA to the masked probability of the wild-type
AA at the same position, keeping all other residues of the context intact:

    LLR(wt → mut | context) = log P(mut | context_masked_at_i)
                            − log P(wt  | context_masked_at_i)

Positive LLR = the PLM prefers the mutant. Negative = the PLM disfavors
it. This is a **ranking heuristic** that correlates with experimental
ΔΔG (Spearman ρ ≈ 0.4–0.5 on ProteinGym) — not a ΔΔG prediction itself.

Implementation:
    - Load `facebook/esm2_t12_35M_UR50D` with a *masked-LM head*
      (AutoModelForMaskedLM), distinct from the mean-pool model used by
      src/pLM_embedder.py.
    - Encode the combined VH+<linker>+VL sequence (or a single chain).
    - For each unique (position, chain) in the candidate set, run ONE
      masked forward pass (mask that one position), softmax the logits,
      record per-AA probabilities.
    - Compute LLR for each candidate from cached per-position
      probabilities — no duplicate forward passes across 19 mutants at
      the same site.

Computational cost: one forward pass per unique (chain, position). For a
full VH+VL antibody (~215 AA), that is ≤ 215 forward passes per antibody
per chain (the enumerator gives ~2.1k candidates but they share positions,
so ~215 passes at most). Takes ~15-40 s on CPU with t12, ~2-5 s on MPS/GPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.mutation.schema import (
    STANDARD_AA,
    AntibodyChain,
    MutationCandidate,
    ScoredMutation,
)


MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
DEFAULT_LINKER = "GGGGSGGGGSGGGGS"  # neutral linker for joint VH+VL encoding


@dataclass
class MaskedMarginalConfig:
    """Runtime config for the masked-marginal scorer."""

    model_name: str = MODEL_NAME
    device: str = "cpu"            # "cpu" | "mps" | "cuda"
    linker: str = DEFAULT_LINKER   # joins VH to VL for joint context
    score_chains_independently: bool = False


# ---------------------------------------------------------------------------


class MaskedMarginalScorer:
    """Run ESM-2 masked-marginal scoring over a candidate set.

    Instances hold the model + tokenizer; reuse across antibodies to
    amortize the ~1 s weight-loading cost.
    """

    def __init__(self, config: MaskedMarginalConfig | None = None, _model=None, _tokenizer=None):
        self.config = config or MaskedMarginalConfig()
        self._model = _model
        self._tokenizer = _tokenizer
        self._aa_to_id: dict[str, int] | None = None
        self._mask_id: int | None = None
        # If the caller injected a tokenizer (tests, or cached load),
        # build the AA index eagerly so .score() works without another load.
        if self._tokenizer is not None:
            self._build_aa_index()

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch required for MaskedMarginalScorer"
            ) from e

        tok = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        model.eval()
        if self.config.device != "cpu":
            try:
                model = model.to(self.config.device)
            except Exception:
                pass  # fall back to CPU silently
        self._model = model
        self._tokenizer = tok
        self._build_aa_index()

    def _build_aa_index(self) -> None:
        if self._tokenizer is None:
            return
        vocab = self._tokenizer.get_vocab()
        aa_to_id: dict[str, int] = {}
        for aa in STANDARD_AA:
            if aa not in vocab:
                raise RuntimeError(f"ESM-2 vocab missing AA {aa!r}; check tokenizer")
            aa_to_id[aa] = vocab[aa]
        self._aa_to_id = aa_to_id
        mask = getattr(self._tokenizer, "mask_token", None)
        if mask is None or mask not in vocab:
            raise RuntimeError("ESM-2 tokenizer has no mask token")
        self._mask_id = vocab[mask]

    # ------------------------------------------------------------------

    def _joint_sequence(self, vh: str, vl: str) -> tuple[str, int, int]:
        """Return (joint_seq, vh_offset, vl_offset) so we can map chain+position."""
        vh = (vh or "").upper()
        vl = (vl or "").upper()
        if not vh:
            raise ValueError("VH sequence required")
        if vl:
            if self.config.score_chains_independently:
                # Returned separately by caller; but keep joint for one-forward-pass mode.
                joint = vh + vl
                return joint, 0, len(vh)
            joint = vh + self.config.linker + vl
            vl_offset = len(vh) + len(self.config.linker)
            return joint, 0, vl_offset
        return vh, 0, 0

    def _chain_position_to_joint(
        self, chain: AntibodyChain, position: int, vh_len: int, vl_offset: int
    ) -> int:
        if chain is AntibodyChain.VH:
            return position
        return vl_offset + position

    # ------------------------------------------------------------------

    def score(
        self,
        vh: str,
        vl: str,
        candidates: list[MutationCandidate],
    ) -> list[ScoredMutation]:
        """Score a candidate list; returns ScoredMutation with LLR only.

        Guardrails are NOT applied here — they are the next pipeline stage.
        """
        self._lazy_load()
        import torch

        joint, vh_offset, vl_offset = self._joint_sequence(vh, vl)
        vh_len = len(vh)

        tok = self._tokenizer.encode(joint, add_special_tokens=True, return_tensors="pt")
        if self.config.device != "cpu":
            try:
                tok = tok.to(self.config.device)
            except Exception:
                pass
        # Special-token offset between raw sequence index and tokenizer index.
        # ESM-2 prepends one BOS token.
        seq_to_tok = 1

        # Group candidates by (chain, position) for shared masked forward pass.
        unique_sites: dict[tuple[AntibodyChain, int], list[MutationCandidate]] = {}
        for c in candidates:
            unique_sites.setdefault((c.chain, c.position), []).append(c)

        # Run ONE forward pass per unique site.
        per_site_probs: dict[tuple[AntibodyChain, int], np.ndarray] = {}
        for (chain, pos), _cands in unique_sites.items():
            joint_pos = self._chain_position_to_joint(chain, pos, vh_len, vl_offset)
            tok_pos = joint_pos + seq_to_tok
            if tok_pos >= tok.shape[1] - 1:
                continue  # past EOS; skip
            masked = tok.clone()
            masked[0, tok_pos] = self._mask_id

            with torch.no_grad():
                logits = self._model(masked).logits  # (1, L, V)
            probs_full = torch.softmax(logits[0, tok_pos], dim=-1).detach().cpu().numpy()

            # Subset to 20 standard AAs, in STANDARD_AA order.
            probs_20 = np.array([probs_full[self._aa_to_id[aa]] for aa in STANDARD_AA])
            probs_20 = probs_20 / probs_20.sum()  # renormalize over 20 AAs
            per_site_probs[(chain, pos)] = probs_20

        # Emit per-candidate LLR.
        aa_idx = {aa: i for i, aa in enumerate(STANDARD_AA)}
        out: list[ScoredMutation] = []
        for c in candidates:
            probs = per_site_probs.get((c.chain, c.position))
            if probs is None:
                continue
            p_wt = float(probs[aa_idx[c.wildtype_aa]])
            p_mt = float(probs[aa_idx[c.mutant_aa]])
            eps = 1e-12
            llr = float(np.log(max(p_mt, eps)) - np.log(max(p_wt, eps)))
            # Rough uncertainty label by magnitude; guardrails refine it.
            mag = abs(llr)
            if mag >= 2.0:
                unc = "high"
            elif mag >= 0.5:
                unc = "medium"
            else:
                unc = "low"
            out.append(
                ScoredMutation(
                    candidate=c,
                    llr=llr,
                    wildtype_prob=p_wt,
                    mutant_prob=p_mt,
                    uncertainty=unc,
                    rationale=(
                        f"ESM-2 masked marginal LLR={llr:+.3f} "
                        f"(P_mt={p_mt:.3f}, P_wt={p_wt:.3f})"
                    ),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Functional convenience — builds a single scorer internally.
# ---------------------------------------------------------------------------


def score_mutations(
    vh: str,
    vl: str,
    candidates: list[MutationCandidate],
    config: MaskedMarginalConfig | None = None,
) -> list[ScoredMutation]:
    """One-call API: instantiate a scorer, score, return."""
    scorer = MaskedMarginalScorer(config=config)
    return scorer.score(vh, vl, candidates)
