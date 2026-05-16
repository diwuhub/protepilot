"""Pluggable antibody-structure predictor.

Tries predictors in order: IgFold → ESMFold → SAbPred (ABodyBuilder2).
If none are installed, returns a StructureInput with source=UNAVAILABLE
and a clear reason in `notes` — so the downstream analyzer can still
emit a StructureMetrics report with `available=False` rather than
crashing.

Predictors are imported lazily so a user with only the analyzer needs
only biopython installed.

Content-addressed cache: every successful prediction writes a PDB to
    runs/structures/{sha256(vh|vl)}/{predictor}.pdb
and subsequent calls return the cached path immediately.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable

from src.structure.schema import StructureInput, StructureSource


DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / "runs" / "structures"


def sequence_pair_hash(vh: str, vl: str = "") -> str:
    """Stable SHA-256 of (VH, VL) — matches the convention used in esm2_features."""
    h = hashlib.sha256()
    h.update(vh.strip().upper().encode("utf-8"))
    h.update(b"|")
    h.update(vl.strip().upper().encode("utf-8"))
    return h.hexdigest()


def _cache_dir(seq_hash: str, cache_root: Path | str | None = None) -> Path:
    root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
    d = root / seq_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_pdb(
    vh: str, vl: str = "", cache_root: Path | str | None = None
) -> Path | None:
    """Return the most-preferred cached PDB if any exists for (vh, vl)."""
    seq_hash = sequence_pair_hash(vh, vl)
    d = _cache_dir(seq_hash, cache_root)
    # Preference order: experimental > igfold > sabpred > esmfold > user.
    for name in (
        "experimental.pdb",
        "igfold.pdb",
        "sabpred.pdb",
        "esmfold.pdb",
        "user.pdb",
    ):
        p = d / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------


def _try_igfold(vh: str, vl: str, cache_dir: Path) -> StructureInput | None:
    """IgFold is the preferred antibody predictor (fast, accurate on CDR-H3)."""
    try:
        from igfold import IgFoldRunner  # type: ignore
    except ImportError:
        return None
    out_pdb = cache_dir / "igfold.pdb"
    if out_pdb.exists():
        return StructureInput(
            pdb_path=str(out_pdb),
            source=StructureSource.IGFOLD,
            confidence=0.85,  # IgFold typical confidence
            sequence_hash=cache_dir.name,
            notes="cached",
        )
    try:
        runner = IgFoldRunner()
        seqs = {"H": vh, "L": vl} if vl else {"H": vh}
        runner.fold(str(out_pdb), sequences=seqs, do_renum=True)
    except Exception as e:
        return StructureInput(
            pdb_path="",
            source=StructureSource.UNAVAILABLE,
            confidence=0.0,
            sequence_hash=cache_dir.name,
            notes=f"IgFold available but failed: {e}",
        )
    return StructureInput(
        pdb_path=str(out_pdb),
        source=StructureSource.IGFOLD,
        confidence=0.85,
        sequence_hash=cache_dir.name,
    )


def _try_esmfold(vh: str, vl: str, cache_dir: Path) -> StructureInput | None:
    """ESMFold fallback — general-purpose, slower on antibodies."""
    try:
        # Many ESMFold Python wrappers exist; we probe the fair-esm path.
        from esm.esmfold.v1.esmfold_web import fold  # type: ignore  # noqa: F401
    except ImportError:
        return None
    out_pdb = cache_dir / "esmfold.pdb"
    if out_pdb.exists():
        return StructureInput(
            pdb_path=str(out_pdb),
            source=StructureSource.ESMFOLD,
            confidence=0.70,
            sequence_hash=cache_dir.name,
            notes="cached",
        )
    # Actual invocation deferred until ESMFold install is standardized
    # in the sequence-feature layer. For now we report unavailable to keep surface contract
    # consistent.
    return StructureInput(
        pdb_path="",
        source=StructureSource.UNAVAILABLE,
        confidence=0.0,
        sequence_hash=cache_dir.name,
        notes="ESMFold module present but wrapper not yet implemented",
    )


def _try_sabpred(vh: str, vl: str, cache_dir: Path) -> StructureInput | None:
    """OPIG SAbPred (ABodyBuilder2). Web-only for free-tier users."""
    # No robust offline Python API at time of Phase 3. Surface contract only.
    return None


_PREDICTORS: list[Callable[[str, str, Path], StructureInput | None]] = [
    _try_igfold,
    _try_esmfold,
    _try_sabpred,
]


# ---------------------------------------------------------------------------


def predict_structure(
    vh: str,
    vl: str = "",
    cache_root: Path | str | None = None,
    prefer_cached: bool = True,
) -> StructureInput:
    """Predict (or look up cached) antibody structure from VH+VL sequences.

    Returns a StructureInput with source=UNAVAILABLE if no predictor is
    installed. Callers should check `input.pdb_path` before opening.
    """
    seq_hash = sequence_pair_hash(vh, vl)
    d = _cache_dir(seq_hash, cache_root)

    if prefer_cached:
        cached = cached_pdb(vh, vl, cache_root)
        if cached is not None:
            return StructureInput(
                pdb_path=str(cached),
                source=_source_from_filename(cached.name),
                confidence=0.85 if cached.name.startswith("igfold") else 1.0 if cached.name.startswith("experimental") else 0.7,
                sequence_hash=seq_hash,
                notes="cached",
            )

    errors: list[str] = []
    for fn in _PREDICTORS:
        result = fn(vh, vl, d)
        if result is None:
            continue
        if result.source is StructureSource.UNAVAILABLE:
            errors.append(result.notes)
            continue
        return result

    reason = "; ".join(errors) if errors else "no predictor installed (tried: igfold, esmfold, sabpred)"
    return StructureInput(
        pdb_path="",
        source=StructureSource.UNAVAILABLE,
        confidence=0.0,
        sequence_hash=seq_hash,
        notes=reason,
    )


def register_user_pdb(
    vh: str,
    vl: str,
    pdb_path: str | Path,
    cache_root: Path | str | None = None,
    source: StructureSource = StructureSource.USER_SUPPLIED,
    confidence: float = 1.0,
    pdb_id: str = "",
) -> StructureInput:
    """Copy a user-supplied PDB into the content-addressed cache."""
    src = Path(pdb_path)
    if not src.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    seq_hash = sequence_pair_hash(vh, vl)
    d = _cache_dir(seq_hash, cache_root)

    if source is StructureSource.EXPERIMENTAL_PDB:
        target_name = "experimental.pdb"
    elif source is StructureSource.USER_SUPPLIED:
        target_name = "user.pdb"
    else:
        target_name = f"{source.value}.pdb"

    target = d / target_name
    if src.resolve() != target.resolve():
        shutil.copy2(src, target)
    return StructureInput(
        pdb_path=str(target),
        source=source,
        confidence=confidence,
        sequence_hash=seq_hash,
        pdb_id=pdb_id,
    )


# ---------------------------------------------------------------------------


def _source_from_filename(name: str) -> StructureSource:
    mapping = {
        "experimental.pdb": StructureSource.EXPERIMENTAL_PDB,
        "igfold.pdb": StructureSource.IGFOLD,
        "esmfold.pdb": StructureSource.ESMFOLD,
        "sabpred.pdb": StructureSource.SABPRED,
        "user.pdb": StructureSource.USER_SUPPLIED,
    }
    return mapping.get(name, StructureSource.USER_SUPPLIED)
