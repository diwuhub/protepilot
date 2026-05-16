"""
Retrain developability XGBoost with real ESM-2 embeddings.
Must be run with src/types.py temporarily renamed to avoid circular import.
"""
import os
import sys
import json
import pickle
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ESM-2 embedder
from pLM_embedder import ESM2Embedder
embedder = ESM2Embedder()
print(f"ESM-2 embedder: mock={embedder.is_mock}")

from esm2_hybrid_encoder import _embed_antibody_batch

# Load merged training data (Jain-137 + merged_wetlab + PROPHET-Ab)
data_path = os.path.join(PROJECT_ROOT, "data", "merged_xgb_training.csv")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} molecules from merged training set")
print(f"  Sources: {df['source'].value_counts().to_dict()}")

# Generate real ESM-2 embeddings
hc_seqs = df["vh"].astype(str).tolist()
lc_seqs = df["vl"].astype(str).tolist()

t0 = time.time()
embeddings = _embed_antibody_batch(hc_seqs, lc_seqs, embedder=embedder)
print(f"ESM-2 embeddings: {embeddings.shape}, took {time.time()-t0:.1f}s")

# Biophysical features
KYTE = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
        "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
        "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}

POS_PKA = {"K": 10.5, "R": 12.5, "H": 6.0}
NEG_PKA = {"D": 3.65, "E": 4.25}

biophys_list = []
for _, row in df.iterrows():
    vh, vl = str(row["vh"]), str(row["vl"])
    seq = vh + vl
    n = max(len(seq), 1)
    gravy = sum(KYTE.get(aa, 0) for aa in seq.upper()) / n
    deam = sum(1 for i in range(n-1) if seq[i].upper()=="N" and seq[i+1].upper() in "GSTD")
    ox = seq.upper().count("M") + seq.upper().count("W")
    acidic = seq.upper().count("D") + seq.upper().count("E")
    basic = seq.upper().count("K") + seq.upper().count("R") + seq.upper().count("H")
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa = ProteinAnalysis(seq.upper().replace("X","A"))
        pI = pa.isoelectric_point()
        mw = pa.molecular_weight() / 1000.0
    except:
        net = basic - acidic
        pI = max(4.0, min(12.0, 7.0 + net * 0.05))
        mw = n * 0.110

    # Extra features (added for stability_slope / smac_rt improvement)
    total_len = len(seq)
    vh_len = len(vh)
    vl_len = len(vl)
    # CDR3 length proxy via conserved WG motif
    vh_upper = vh.upper()
    wg_pos = vh_upper.rfind("WG")
    cdr3_proxy = 12  # default
    if wg_pos > 80:
        c_pos = vh_upper.rfind("C", max(0, wg_pos - 35), wg_pos)
        if c_pos > 0:
            cdr3_proxy = wg_pos - c_pos - 1
    # Net charge at pH 7.4
    charge_74 = 0.0
    for aa in seq.upper():
        if aa in POS_PKA:
            charge_74 += 1.0 / (1.0 + 10.0**(7.4 - POS_PKA[aa]))
        elif aa in NEG_PKA:
            charge_74 -= 1.0 / (1.0 + 10.0**(NEG_PKA[aa] - 7.4))
    charge_74 += 1.0 / (1.0 + 10.0**(7.4 - 9.0))
    charge_74 -= 1.0 / (1.0 + 10.0**(2.3 - 7.4))
    # Aromatic fraction
    aromatic = sum(1 for aa in seq.upper() if aa in "FWY") / n

    biophys_list.append([pI, mw, deam, ox, acidic, basic, gravy,
                         total_len, vh_len, vl_len, cdr3_proxy,
                         charge_74, aromatic])

biophys = np.array(biophys_list)
BIOPHYS_DIM = biophys.shape[1]  # 13
print(f"Biophysical features: {biophys.shape} (7 original + 6 new)")

# Combine: ESM-2 (960) + biophys (13) = 973
X = np.hstack([embeddings, biophys])
FEAT_DIM = X.shape[1]
print(f"Combined features: {X.shape}")

# Target columns (standardized names from merged dataset)
target_map = {
    "fab_tm": "fab_tm",
    "hic_rt": "hic_rt",
    "acsins": "acsins",
    "psr": "psr",
    "stability_slope": "stability_slope",
    "titer": "titer",
    "smac_rt": "smac_rt",
}

import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold

output_dir = os.path.join(PROJECT_ROOT, "models", "developability")
os.makedirs(output_dir, exist_ok=True)

results = {}
seed = 42

# Default hyperparameters
DEFAULT_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=seed,
    tree_method="hist",
)

# Target-specific overrides for weak targets
TARGET_OVERRIDES = {
    "smac_rt": {
        "params": dict(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.3, min_child_weight=10,
            reg_alpha=2.0, reg_lambda=10.0, random_state=seed,
            tree_method="hist",
        ),
        # PROPHET-Ab SMAC values are on incompatible scale (mean 2.9 vs Jain 10.0)
        "source_filter": "jain137",
    },
    "stability_slope": {
        "params": DEFAULT_PARAMS,  # baseline params performed best
        "outlier_sigma": 3,        # remove 3-sigma outliers
        "log_transform": True,     # log1p to spread compressed distribution
    },
}

for target_key, col_name in target_map.items():
    if col_name not in df.columns:
        print(f"  {target_key}: column '{col_name}' NOT FOUND")
        continue

    overrides = TARGET_OVERRIDES.get(target_key, {})
    params = overrides.get("params", DEFAULT_PARAMS)

    y = df[col_name].values.astype(float)
    valid = ~np.isnan(y)

    # Source filtering (e.g., smac_rt uses only Jain-137)
    if "source_filter" in overrides:
        src = overrides["source_filter"]
        valid = valid & (df["source"] == src).values
        print(f"  {target_key}: filtered to source='{src}'")

    X_valid = X[valid]
    y_valid = y[valid]

    if len(y_valid) < 10:
        print(f"  {target_key}: only {len(y_valid)} valid samples, skipping")
        continue

    # Outlier removal
    if overrides.get("outlier_sigma"):
        sigma = overrides["outlier_sigma"]
        mean_y, std_y = y_valid.mean(), y_valid.std()
        inlier = np.abs(y_valid - mean_y) < sigma * std_y
        n_removed = (~inlier).sum()
        X_valid = X_valid[inlier]
        y_valid = y_valid[inlier]
        if n_removed > 0:
            print(f"  {target_key}: removed {n_removed} outlier(s) (>{sigma}-sigma)")

    # Log transform
    y_train_target = y_valid
    y_min_offset = 0.0
    if overrides.get("log_transform"):
        y_min_offset = float(y_valid.min() - 0.001)
        y_train_target = np.log1p(y_valid - y_min_offset)
        print(f"  {target_key}: applied log1p transform")

    X_train, X_val, y_train, y_val = train_test_split(
        X_valid, y_train_target, test_size=0.2, random_state=seed
    )
    # Keep original-scale val for evaluation
    _, y_val_orig_split = train_test_split(
        y_valid, y_valid, test_size=0.2, random_state=seed
    )

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              verbose=False)

    y_pred = model.predict(X_val)

    # Evaluate in original space
    if overrides.get("log_transform"):
        y_pred_orig = np.expm1(y_pred) + y_min_offset
        y_val_eval = np.expm1(y_val) + y_min_offset
    else:
        y_pred_orig = y_pred
        y_val_eval = y_val

    r, _ = sp_stats.pearsonr(y_val_eval, y_pred_orig)
    rho, _ = sp_stats.spearmanr(y_val_eval, y_pred_orig)
    rmse = np.sqrt(np.mean((y_val_eval - y_pred_orig) ** 2))
    gate = "PASS" if abs(rho) > 0.15 else "FAIL"

    # Save model
    model_path = os.path.join(output_dir, f"xgb_{target_key}.pkl")
    model_meta = {
        "model": model, "feature_dim": FEAT_DIM, "embedding_mode": "esm2",
    }
    if overrides.get("log_transform"):
        model_meta["target_transform"] = "log1p"
        model_meta["target_min_offset"] = y_min_offset
    if "source_filter" in overrides:
        model_meta["training_source"] = overrides["source_filter"]
    with open(model_path, "wb") as f:
        pickle.dump(model_meta, f)

    results[target_key] = {
        "n": int(len(y_valid)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "pearson_r": round(float(r), 4),
        "spearman_rho": round(float(rho), 4),
        "rmse": round(float(rmse), 4),
        "gate": gate,
        "model_path": model_path,
        "feature_dim": FEAT_DIM,
        "embedding_mode": "esm2",
    }
    print(f"  {target_key}: r={r:.3f}, rho={rho:.3f}, rmse={rmse:.4f}, gate={gate}")

# Save metadata
meta = {
    "format_version": "3.1",
    "model_type": "xgboost",
    "feature_dim": FEAT_DIM,
    "embedding_dim": embeddings.shape[1],
    "biophys_dim": BIOPHYS_DIM,
    "embedding_mode": "esm2",
    "n_molecules": len(df),
    "data_source": data_path,
    "seed": seed,
    "targets": results,
}
meta_path = os.path.join(output_dir, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nMetadata saved: {meta_path}")
print("Done!")
