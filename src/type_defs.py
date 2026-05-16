"""
type_defs.py  ·  ProtePilot — Shared Type Definitions
==================================================
Platform-wide types that are imported by multiple modules.

Placing shared types here avoids circular imports and makes the
dependency graph cleaner:  any module can import from src.type_defs
without pulling in heavy analysis logic.

Currently contains:
- MoleculeClass enum (used by 8+ modules)

Design decision: molecule_classifier.py re-exports MoleculeClass
so that existing `from src.molecule_classifier import MoleculeClass`
imports continue to work (zero-breakage migration).

Author  : Di (ProtePilot)
"""

from __future__ import annotations

from enum import Enum


class MoleculeClass(str, Enum):
    """
    Biologics molecule classification.

    Each class implies a distinct analysis pathway:
    - Feature selection (which biophysical descriptors are relevant)
    - Model routing   (which ML/physics models to apply)
    - Risk weights    (how to weight aggregation vs expression vs PK)
    - Validation plan (which assays to recommend)
    """
    CANONICAL_MAB       = "canonical_mab"
    BISPECIFIC          = "bispecific"
    FC_FUSION           = "fc_fusion"
    ADC                 = "adc"
    SINGLE_DOMAIN       = "single_domain"
    PEPTIDE             = "peptide"
    FUSION_PROTEIN      = "fusion_protein"
    ENGINEERED_SCAFFOLD = "engineered_scaffold"
    UNKNOWN             = "unknown"

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        _names = {
            "canonical_mab": "Canonical mAb (IgG)",
            "bispecific": "Bispecific Antibody",
            "fc_fusion": "Fc-Fusion Protein",
            "adc": "Antibody-Drug Conjugate (ADC)",
            "single_domain": "Single-Domain Antibody (VHH/Nanobody)",
            "peptide": "Therapeutic Peptide",
            "fusion_protein": "Fusion Protein",
            "engineered_scaffold": "Engineered Scaffold",
            "unknown": "Unclassified Biologic",
        }
        return _names.get(self.value, self.value)

    @property
    def has_fc_region(self) -> bool:
        """Whether this format typically contains an Fc domain."""
        return self.value in (
            "canonical_mab", "bispecific", "fc_fusion", "adc",
        )

    @property
    def expects_glycosylation(self) -> bool:
        """Whether N297-type Fc glycosylation is expected."""
        return self.has_fc_region

    @property
    def is_multi_chain(self) -> bool:
        """Whether this format involves multiple distinct polypeptide chains."""
        return self.value in (
            "canonical_mab", "bispecific", "fc_fusion", "adc",
        )

    @property
    def is_mab_like(self) -> bool:
        """Whether this format is structurally mAb-like (IgG scaffold with disulfide bonds).

        Used by analytical_twin for disulfide bond estimation and glycosylation assumptions.
        ADCs use the mAb scaffold; Fc-fusions share the Fc but differ in Fab.
        """
        return self.value in (
            "canonical_mab", "bispecific", "adc",
        )

    @property
    def expected_disulfide_bonds(self) -> int:
        """Expected disulfide bonds for this molecule class.

        Standard IgG1: 4 inter-chain + 12 intra-chain = 16.
        Fc-fusion: ~6-8 (Fc inter/intra only, no Fab).
        Single-domain: 1-2 (canonical VHH fold).
        """
        _ds_map = {
            "canonical_mab": 16, "bispecific": 16, "adc": 16,
            "fc_fusion": 8, "single_domain": 1, "peptide": 0,
            "fusion_protein": 0, "engineered_scaffold": 0, "unknown": 0,
        }
        return _ds_map.get(self.value, 0)
