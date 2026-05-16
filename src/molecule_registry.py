"""
molecule_registry.py  ·  ProtePilot — Central Molecule-Class Configuration Registry
=====================================================================================
Single source of truth for all molecule-class-specific behavior.

Previously, this configuration was spread across 5+ files in if/elif branches.
Now every module reads from this registry instead of hard-coding format-specific
logic. This makes behavior testable, auditable, and extensible.

Registry contents per molecule class:
  - display_name:          Human-readable name
  - risk_weights:          5+ dimension weights for composite scoring
  - immuno_floor:          Floor immunogenicity score (0.0 = no floor)
  - immuno_floor_reason:   Evidence text for the floor
  - confidence_defaults:   Per-dimension confidence when data is missing
  - recommendation_suffix: Appended to recommendation detail text
  - validation_caveat:     Caveat text for validation plan section
  - is_non_canonical:      True for any format that is NOT canonical mAb
  - extra_dimensions:      Additional risk dimensions beyond the standard 5

Usage:
    from src.molecule_registry import MOLECULE_REGISTRY, get_config

    config = get_config("bispecific")
    config["risk_weights"]              # {"aggregation": 0.25, ...}
    config["recommendation_suffix"]     # " For this bispecific format: ..."
    config["immuno_floor"]              # 0.0 (no special floor for bispecific)

To add a new molecule class:
    1. Add an entry to MOLECULE_REGISTRY below
    2. Add the class to MoleculeClass enum in molecule_classifier.py
    3. Run: pytest -m governance  (verifies registry completeness)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
#  The Registry
# ═══════════════════════════════════════════════════════════════════════

MOLECULE_REGISTRY: Dict[str, Dict[str, Any]] = {

    "canonical_mab": {
        "display_name": "Monoclonal Antibody (IgG)",
        "is_non_canonical": False,
        "risk_weights": {
            "aggregation": 0.30, "stability": 0.25, "viscosity": 0.20,
            "expression": 0.15, "immunogenicity": 0.10,
        },
        "extra_dimensions": [],
        "immuno_floor": 0.0,
        "immuno_floor_reason": "",
        "confidence_defaults": {
            "aggregation": "High", "stability": "High", "viscosity": "Medium",
            "expression": "Medium", "immunogenicity": "Medium",
        },
        "recommendation_suffix": "",
        "validation_caveat": (
            "Standard IgG developability benchmarks applicable. "
            "Predictions validated against Jain-137 and internal datasets."
        ),
    },

    "bispecific": {
        "display_name": "Bispecific Antibody",
        "is_non_canonical": True,
        "risk_weights": {
            "aggregation": 0.25, "stability": 0.20, "viscosity": 0.15,
            "expression": 0.15, "immunogenicity": 0.10,
            "species_purity": 0.15,
        },
        "extra_dimensions": [
            {
                "name": "species_purity",
                "display_name": "Species Purity",
                "default_score": 0.3,
                "default_evidence": ["Bispecific format requires homodimer separation assessment"],
                "default_explanation": (
                    "Bispecific antibodies produce AA/AB/BB species mixtures. "
                    "Homodimer removal is critical for efficacy."
                ),
                "source": "Bispecific Engine",
                "confidence": "Low",
            },
        ],
        "immuno_floor": 0.0,
        "immuno_floor_reason": "",
        "confidence_defaults": {
            "aggregation": "Medium", "stability": "Medium", "viscosity": "Medium",
            "expression": "Medium", "immunogenicity": "Medium",
            "species_purity": "Low",
        },
        "recommendation_suffix": (
            " Recommend format-specific characterization including species-control "
            "strategy, homodimer quantification, and bispecific-specific potency assays."
        ),
        "validation_caveat": (
            "Bispecific-specific predictions (species purity, assembly risk) are "
            "based on limited benchmark data. Dual-target binding should be confirmed "
            "experimentally by SPR/BLI."
        ),
    },

    "fc_fusion": {
        "display_name": "Fc-Fusion Protein",
        "is_non_canonical": True,
        "risk_weights": {
            "aggregation": 0.30, "stability": 0.20, "viscosity": 0.15,
            "expression": 0.25, "immunogenicity": 0.10,
        },
        "extra_dimensions": [],
        "immuno_floor": 0.0,
        "immuno_floor_reason": "",
        "confidence_defaults": {
            "aggregation": "Medium", "stability": "Medium", "viscosity": "Medium",
            "expression": "Medium", "immunogenicity": "Medium",
        },
        "recommendation_suffix": (
            " Recommend Fc-fusion linker stability assessment and half-life "
            "extension validation with FcRn binding confirmation."
        ),
        "validation_caveat": (
            "Fc-fusion proteins have distinct linker clipping and aggregation profiles. "
            "Predictions trained on IgG benchmarks; format-specific validation recommended."
        ),
    },

    "adc": {
        "display_name": "Antibody-Drug Conjugate",
        "is_non_canonical": True,
        "risk_weights": {
            "aggregation": 0.20, "stability": 0.25, "viscosity": 0.10,
            "expression": 0.15, "immunogenicity": 0.15,
            "conjugation": 0.15,
        },
        "extra_dimensions": [
            {
                "name": "conjugation",
                "display_name": "Conjugation Integrity",
                "default_score": 0.3,
                "default_evidence": ["ADC requires DAR distribution and payload stability monitoring"],
                "default_explanation": (
                    "ADC conjugation chemistry affects DAR distribution, "
                    "payload stability, and pharmacokinetic behavior."
                ),
                "source": "Heuristic",
                "confidence": "Low",
            },
        ],
        "immuno_floor": 0.0,
        "immuno_floor_reason": "",
        "confidence_defaults": {
            "aggregation": "Medium", "stability": "Medium", "viscosity": "Medium",
            "expression": "Medium", "immunogenicity": "Medium",
            "conjugation": "Low",
        },
        "recommendation_suffix": (
            " Recommend conjugation chemistry DAR consistency, payload stability, "
            "and linker-drug pharmacokinetic characterization."
        ),
        "validation_caveat": (
            "ADC developability depends heavily on conjugation chemistry, linker type, "
            "and payload properties — not captured by sequence alone."
        ),
    },

    "single_domain": {
        "display_name": "Single-Domain Antibody (Nanobody/VHH)",
        "is_non_canonical": True,
        "risk_weights": {
            "aggregation": 0.35, "stability": 0.30, "viscosity": 0.05,
            "expression": 0.20, "immunogenicity": 0.10,
        },
        "extra_dimensions": [],
        "immuno_floor": 0.0,
        "immuno_floor_reason": "",
        "confidence_defaults": {
            "aggregation": "Medium", "stability": "Medium", "viscosity": "Low",
            "expression": "Medium", "immunogenicity": "Low",
        },
        "recommendation_suffix": (
            " Recommend aggregation-prone format monitoring, thermal stability "
            "confirmation, and half-life extension strategy assessment."
        ),
        "validation_caveat": (
            "Single-domain antibodies have distinct aggregation and renal clearance "
            "profiles. Benchmark data is limited; thermal stability should be confirmed."
        ),
    },

    "peptide": {
        "display_name": "Peptide Therapeutic",
        "is_non_canonical": True,
        "risk_weights": {
            "aggregation": 0.10, "stability": 0.40, "viscosity": 0.05,
            "expression": 0.10, "immunogenicity": 0.35,
        },
        "extra_dimensions": [],
        "immuno_floor": 0.30,
        "immuno_floor_reason": "Small peptide — limited T-cell epitope masking",
        "confidence_defaults": {
            "aggregation": "Low", "stability": "Medium", "viscosity": "Low",
            "expression": "Low", "immunogenicity": "Medium",
        },
        "recommendation_suffix": (
            " Recommend peptide-specific stability profiling, protease susceptibility, "
            "and formulation compatibility for subcutaneous delivery."
        ),
        "validation_caveat": (
            "Peptide predictions are based on amino-acid composition only; "
            "chemical modifications (cyclization, PEGylation, non-natural AAs) "
            "are not modelled."
        ),
    },

    "fusion_protein": {
        "display_name": "Fusion Protein",
        "is_non_canonical": True,
        "risk_weights": {
            "aggregation": 0.30, "stability": 0.25, "viscosity": 0.15,
            "expression": 0.20, "immunogenicity": 0.10,
        },
        "extra_dimensions": [],
        "immuno_floor": 0.0,
        "immuno_floor_reason": "",
        "confidence_defaults": {
            "aggregation": "Medium", "stability": "Medium", "viscosity": "Medium",
            "expression": "Medium", "immunogenicity": "Medium",
        },
        "recommendation_suffix": (
            " Recommend domain interface stability, linker integrity, and "
            "functional activity of each fusion partner independently."
        ),
        "validation_caveat": (
            "Fusion protein predictions assume independent domain behavior; "
            "domain-domain interactions may alter aggregation and stability."
        ),
    },

    "engineered_scaffold": {
        "display_name": "Engineered Scaffold",
        "is_non_canonical": True,
        "risk_weights": {
            "aggregation": 0.30, "stability": 0.25, "viscosity": 0.10,
            "expression": 0.15, "immunogenicity": 0.20,
        },
        "extra_dimensions": [],
        "immuno_floor": 0.35,
        "immuno_floor_reason": "Non-human scaffold framework — inherently higher ADA risk",
        "confidence_defaults": {
            "aggregation": "Low", "stability": "Low", "viscosity": "Low",
            "expression": "Low", "immunogenicity": "Medium",
        },
        "recommendation_suffix": (
            " Recommend scaffold-specific immunogenicity assessment and "
            "comparability to published safety data for this format."
        ),
        "validation_caveat": (
            "Engineered scaffolds (DARPins, affibodies, etc.) have no IgG benchmark. "
            "All predictions carry low confidence; scaffold-specific data is essential."
        ),
    },

    "unknown": {
        "display_name": "Unknown / Unclassified",
        "is_non_canonical": True,
        "risk_weights": {
            "aggregation": 0.30, "stability": 0.25, "viscosity": 0.15,
            "expression": 0.15, "immunogenicity": 0.15,
        },
        "extra_dimensions": [],
        "immuno_floor": 0.0,
        "immuno_floor_reason": "",
        "confidence_defaults": {
            "aggregation": "Low", "stability": "Low", "viscosity": "Low",
            "expression": "Low", "immunogenicity": "Low",
        },
        "recommendation_suffix": (
            " Format identification is a prerequisite — all predictions assume "
            "canonical IgG behavior and may not apply. Confirm molecule format "
            "experimentally before relying on risk assessments."
        ),
        "validation_caveat": (
            "Molecule format could not be determined. All predictions assume "
            "canonical IgG behavior and may be unreliable."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Evidence Tier Configuration
# ═══════════════════════════════════════════════════════════════════════

EVIDENCE_TIERS: Dict[str, Dict[str, Any]] = {
    "primary": {
        "display_name": "Primary Evidence",
        "multiplier": 1.0,
        "narrative": "Sequence-derived, experimentally validated",
        "confidence": "High",
    },
    "supporting": {
        "display_name": "Supporting Evidence (predicted)",
        "multiplier": 0.8,
        "narrative": "Predicted by ML or heuristic model",
        "confidence": "Medium",
    },
    "simulated": {
        "display_name": "Simulated / Supportive Only",
        "multiplier": 0.5,
        "narrative": "Virtual QC or extrapolated from limited data",
        "confidence": "Low",
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def get_config(molecule_class: str) -> Dict[str, Any]:
    """Get the full configuration for a molecule class.

    Falls back to 'unknown' for unrecognized classes.

    >>> config = get_config("bispecific")
    >>> config["risk_weights"]["species_purity"]
    0.15
    """
    return MOLECULE_REGISTRY.get(molecule_class, MOLECULE_REGISTRY["unknown"])


def get_risk_weights(molecule_class: str) -> Dict[str, float]:
    """Get risk weights for a molecule class (convenience alias)."""
    return get_config(molecule_class)["risk_weights"]


def get_recommendation_suffix(molecule_class: str) -> str:
    """Get the recommendation suffix text for a molecule class."""
    return get_config(molecule_class)["recommendation_suffix"]


def get_immuno_floor(molecule_class: str) -> Tuple[float, str]:
    """Get the immunogenicity floor score and reason for a molecule class.

    Returns (0.0, "") if no floor applies.
    """
    cfg = get_config(molecule_class)
    return cfg["immuno_floor"], cfg["immuno_floor_reason"]


def get_validation_caveat(molecule_class: str) -> str:
    """Get the validation caveat text for a molecule class."""
    return get_config(molecule_class)["validation_caveat"]


def is_non_canonical(molecule_class: str) -> bool:
    """Check if a molecule class is non-canonical (not standard mAb)."""
    return get_config(molecule_class)["is_non_canonical"]


def get_extra_dimensions(molecule_class: str) -> List[Dict[str, Any]]:
    """Get extra risk dimensions for a molecule class (e.g., species_purity for bispecific)."""
    return get_config(molecule_class)["extra_dimensions"]


def all_molecule_classes() -> List[str]:
    """Return all registered molecule class names."""
    return list(MOLECULE_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    """Validate registry integrity."""
    required_keys = {
        "display_name", "is_non_canonical", "risk_weights", "extra_dimensions",
        "immuno_floor", "immuno_floor_reason", "confidence_defaults",
        "recommendation_suffix", "validation_caveat",
    }
    standard_dims = {"aggregation", "stability", "viscosity", "expression", "immunogenicity"}

    for cls_name, cfg in MOLECULE_REGISTRY.items():
        missing = required_keys - set(cfg.keys())
        assert not missing, f"{cls_name}: missing keys {missing}"

        # Risk weights must include at least the standard 5 dimensions
        w = cfg["risk_weights"]
        assert standard_dims.issubset(w.keys()), (
            f"{cls_name}: missing standard risk dimensions: {standard_dims - set(w.keys())}"
        )

        # Weights must sum to ~1.0 (±0.05 tolerance)
        total = sum(w.values())
        assert 0.95 <= total <= 1.05, f"{cls_name}: risk weights sum={total:.3f} (expected ~1.0)"

        # Extra dimensions must appear in risk_weights
        for dim in cfg["extra_dimensions"]:
            assert dim["name"] in w, f"{cls_name}: extra dim '{dim['name']}' not in risk_weights"

        # Confidence defaults should cover all risk weight keys
        for dim_name in w:
            assert dim_name in cfg["confidence_defaults"], (
                f"{cls_name}: no confidence_default for dimension '{dim_name}'"
            )

    # Verify canonical_mab is the only non-non-canonical
    canonical = [k for k, v in MOLECULE_REGISTRY.items() if not v["is_non_canonical"]]
    assert canonical == ["canonical_mab"], f"Only canonical_mab should be canonical, got {canonical}"

    print(f"molecule_registry selftest PASS ({len(MOLECULE_REGISTRY)} classes)")
    return True


if __name__ == "__main__":
    _selftest()
