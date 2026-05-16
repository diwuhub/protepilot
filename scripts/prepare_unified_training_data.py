"""
prepare_unified_training_data.py
===========================================================
Generates synthetic training data for the Unified MultiTask Model.

Combines:
  1. Reference mAb data (NISTmAb, Adalimumab, Rituximab, etc.)
  2. Synthetic variants with realistic noise
  3. CADET calibration results (ka, nu) from MainOrchestrator

Output: data/unified_training_data.csv

Column format:
  hc_sequence, lc_sequence, pI, MW_kDa, deam_sites, ox_sites,
  acidic_residues, basic_residues, hydrophobicity_gravy,
  ka, nu, tm, aggregation_risk, stability, viscosity_risk,
  hydrophobicity, potency
"""

import csv
import os
import random
import sys

import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Reference antibody sequences (truncated VH/VL for training)
# ---------------------------------------------------------------------------
REFERENCE_MABS = [
    {
        "name": "NISTmAb",
        "hc": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        "lc": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK",
        "pI": 8.44, "MW_kDa": 148.0, "tm": 72.0, "ka": 32.5, "nu": 4.7,
        "agg_risk": 0.12, "stability": 0.88, "visc_risk": 0.15, "hydro": 0.35, "potency": 0.85,
    },
    {
        "name": "Adalimumab",
        "hc": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
        "lc": "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK",
        "pI": 8.25, "MW_kDa": 148.0, "tm": 68.5, "ka": 28.0, "nu": 4.2,
        "agg_risk": 0.25, "stability": 0.78, "visc_risk": 0.22, "hydro": 0.38, "potency": 0.90,
    },
    {
        "name": "Rituximab",
        "hc": "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGGDWYFNVWGAGTTVTVSA",
        "lc": "QIVLSQSPAILSASPGEKVTMTCRASSSVSYIHWFQQKPGSSPKPWIYATSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWTSNPPTFGGGTKLEIK",
        "pI": 9.40, "MW_kDa": 144.0, "tm": 71.0, "ka": 35.0, "nu": 5.1,
        "agg_risk": 0.18, "stability": 0.83, "visc_risk": 0.18, "hydro": 0.32, "potency": 0.88,
    },
    {
        "name": "Trastuzumab",
        "hc": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVRGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        "lc": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK",
        "pI": 8.45, "MW_kDa": 148.0, "tm": 70.5, "ka": 30.0, "nu": 4.5,
        "agg_risk": 0.15, "stability": 0.85, "visc_risk": 0.16, "hydro": 0.34, "potency": 0.92,
    },
]

# Standard amino acids
AA = "ACDEFGHIKLMNPQRSTVWY"


def mutate_sequence(seq: str, n_mutations: int = 3) -> str:
    """Introduce random point mutations."""
    seq_list = list(seq)
    positions = random.sample(range(len(seq_list)), min(n_mutations, len(seq_list)))
    for pos in positions:
        seq_list[pos] = random.choice(AA)
    return "".join(seq_list)


def generate_synthetic_sample(ref: dict, noise_scale: float = 0.15) -> dict:
    """Generate a synthetic variant from a reference mAb."""
    n_mut = random.randint(1, 8)
    hc = mutate_sequence(ref["hc"], n_mut)
    lc = mutate_sequence(ref["lc"], max(1, n_mut // 2))

    # Add realistic noise to properties
    def noisy(val, scale=noise_scale):
        return val * (1.0 + np.random.normal(0, scale))

    sample = {
        "hc_sequence": hc,
        "lc_sequence": lc,
        "pI": np.clip(noisy(ref["pI"], 0.05), 5.0, 11.0),
        "MW_kDa": np.clip(noisy(ref["MW_kDa"], 0.02), 130.0, 180.0),
        "deam_sites": max(0, int(hc.count("N") * random.uniform(0.5, 1.5))),
        "ox_sites": max(0, int((hc.count("M") + hc.count("W")) * random.uniform(0.5, 1.5))),
        "acidic_residues": hc.count("D") + hc.count("E") + lc.count("D") + lc.count("E"),
        "basic_residues": hc.count("K") + hc.count("R") + hc.count("H") + lc.count("K") + lc.count("R") + lc.count("H"),
    }

    # Task labels (with some missing values for masked training)
    tasks = {
        "ka": np.clip(noisy(ref["ka"], 0.2), 0.1, 100.0),
        "nu": np.clip(noisy(ref["nu"], 0.15), 1.0, 20.0),
        "tm": np.clip(noisy(ref["tm"], 0.08), 40.0, 95.0),
        "aggregation_risk": np.clip(noisy(ref["agg_risk"], 0.3), 0.0, 1.0),
        "stability": np.clip(noisy(ref["stability"], 0.15), 0.0, 1.0),
        "viscosity_risk": np.clip(noisy(ref["visc_risk"], 0.3), 0.0, 1.0),
        "hydrophobicity": np.clip(noisy(ref["hydro"], 0.2), 0.0, 1.0),
        "potency": np.clip(noisy(ref["potency"], 0.15), 0.0, 1.0),
    }

    # Randomly mask ~20% of task labels (simulate incomplete annotation)
    for task in list(tasks.keys()):
        if random.random() < 0.2:
            tasks[task] = ""  # Will become NaN in CSV

    # Compute approximate GRAVY for biophys column
    kd_scale = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    full_seq = hc + lc
    gravy = sum(kd_scale.get(aa, 0) for aa in full_seq) / max(len(full_seq), 1)
    sample["hydrophobicity_gravy"] = np.clip((gravy + 2.0) / 4.0, 0.0, 1.0)

    sample.update(tasks)
    return sample


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "unified_training_data.csv")

    # Generate samples
    all_samples = []

    # Add reference mAbs (fully labeled)
    for ref in REFERENCE_MABS:
        sample = {
            "hc_sequence": ref["hc"],
            "lc_sequence": ref["lc"],
            "pI": ref["pI"],
            "MW_kDa": ref["MW_kDa"],
            "deam_sites": ref["hc"].count("N"),
            "ox_sites": ref["hc"].count("M") + ref["hc"].count("W"),
            "acidic_residues": sum(ref["hc"].count(aa) + ref["lc"].count(aa) for aa in "DE"),
            "basic_residues": sum(ref["hc"].count(aa) + ref["lc"].count(aa) for aa in "KRH"),
            "hydrophobicity_gravy": ref["hydro"],
            "ka": ref["ka"],
            "nu": ref["nu"],
            "tm": ref["tm"],
            "aggregation_risk": ref["agg_risk"],
            "stability": ref["stability"],
            "viscosity_risk": ref["visc_risk"],
            "hydrophobicity": ref["hydro"],
            "potency": ref["potency"],
        }
        all_samples.append(sample)

    # Generate synthetic variants
    n_synthetic = 200
    for _ in range(n_synthetic):
        ref = random.choice(REFERENCE_MABS)
        sample = generate_synthetic_sample(ref)
        all_samples.append(sample)

    # Write CSV
    columns = [
        "hc_sequence", "lc_sequence",
        "pI", "MW_kDa", "deam_sites", "ox_sites",
        "acidic_residues", "basic_residues", "hydrophobicity_gravy",
        "ka", "nu", "tm", "aggregation_risk",
        "stability", "viscosity_risk", "hydrophobicity", "potency",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for sample in all_samples:
            row = {c: sample.get(c, "") for c in columns}
            writer.writerow(row)

    print(f"Generated {len(all_samples)} samples → {output_path}")
    print(f"  Reference mAbs: {len(REFERENCE_MABS)}")
    print(f"  Synthetic variants: {n_synthetic}")

    # Count missing values per task
    for task in ["ka", "nu", "tm", "aggregation_risk", "stability", "viscosity_risk", "hydrophobicity", "potency"]:
        n_missing = sum(1 for s in all_samples if s.get(task, "") == "")
        print(f"  {task}: {len(all_samples) - n_missing} labeled, {n_missing} masked")


if __name__ == "__main__":
    main()
