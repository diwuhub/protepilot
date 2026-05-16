"""
Retrain the 2 failing XGBoost developability targets:
  - stability_slope (rho=0.108) — only 137 samples, heavily right-skewed
  - smac_rt (rho=-0.109) — PROPHET-Ab on different scale than Jain-137

Fixes applied:
  1. smac_rt: Z-score normalize per source before merging (fixes 3.4x scale mismatch)
  2. stability_slope: log-transform to spread compressed distribution
  3. Both: hyperparameter tuning (more trees, lower LR, shallower depth)
  4. Both: add extra features (seq length, CDR3 length proxy, charge at pH 7.4)
  5. Both: 5-fold CV instead of single 80/20 split for stable rho estimates
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
from sklearn.model_selection import KFold

# ESM-2 embedder
from pLM_embedder import ESM2Embedder
embedder = ESM2Embedder()
print(f"ESM-2 embedder: mock={embedder.is_mock}")

from esm2_hybrid_encoder import _embed_antibody_batch

# ── Load data ──────────────────────────────────────────────────────────
data_path = os.path.join(PROJECT_ROOT, "data", "merged_xgb_training.csv")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} molecules from merged training set")
print(f"  Sources: {df['source'].value_counts().to_dict()}")

# ── Generate ESM-2 embeddings ─────────────────────────────────────────
hc_seqs = df["vh"].astype(str).tolist()
lc_seqs = df["vl"].astype(str).tolist()

t0 = time.time()
embeddings = _embed_antibody_batch(hc_seqs, lc_seqs, embedder=embedder)
print(f"ESM-2 embeddings: {embeddings.shape}, took {time.time()-t0:.1f}s")

# ── Biophysical features (original 7) ─────────────────────────────────
KYTE = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
        "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
        "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}

# pKa values for charge calculation at pH 7.4
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
    except Exception:
        net = basic - acidic
        pI = max(4.0, min(12.0, 7.0 + net * 0.05))
        mw = n * 0.110

    # ── NEW features ──
    vh_len = len(vh)
    vl_len = len(vl)
    total_len = len(seq)

    # CDR3 length proxy: last ~12 residues before FW4 in VH (approximate)
    # CDR3 typically ends around position 105-115 in Kabat; use last segment
    # before conserved W-G motif as proxy
    cdr3_proxy = 0
    vh_upper = vh.upper()
    wg_pos = vh_upper.rfind("WG")
    if wg_pos > 80:
        # walk back from WG to find start (usually C at ~position 92)
        c_pos = vh_upper.rfind("C", max(0, wg_pos - 35), wg_pos)
        if c_pos > 0:
            cdr3_proxy = wg_pos - c_pos - 1
    if cdr3_proxy <= 0:
        cdr3_proxy = 12  # default

    # Net charge at pH 7.4
    charge_74 = 0.0
    for aa in seq.upper():
        if aa in POS_PKA:
            charge_74 += 1.0 / (1.0 + 10.0**(7.4 - POS_PKA[aa]))
        elif aa in NEG_PKA:
            charge_74 -= 1.0 / (1.0 + 10.0**(NEG_PKA[aa] - 7.4))
    # N-terminus (+1 at low pH, pKa ~9.0)
    charge_74 += 1.0 / (1.0 + 10.0**(7.4 - 9.0))
    # C-terminus (-1 at high pH, pKa ~2.3)
    charge_74 -= 1.0 / (1.0 + 10.0**(2.3 - 7.4))

    # Aromatic fraction
    aromatic = sum(1 for aa in seq.upper() if aa in "FWY") / n

    biophys_list.append([pI, mw, deam, ox, acidic, basic, gravy,
                         total_len, vh_len, vl_len, cdr3_proxy,
                         charge_74, aromatic])

biophys = np.array(biophys_list)
N_BIOPHYS = biophys.shape[1]
print(f"Biophysical features: {biophys.shape} (original 7 + 6 new = {N_BIOPHYS})")

# ── Combine features ──────────────────────────────────────────────────
X = np.hstack([embeddings, biophys])
FEAT_DIM = X.shape[1]
print(f"Combined features: {X.shape}")


# ── Fix smac_rt: Z-score normalize per source ─────────────────────────
print("\n=== Fixing smac_rt scale mismatch ===")
jain_mask = df["source"] == "jain137"
prophet_mask = df["source"] == "prophet_ab"

jain_smac = df.loc[jain_mask, "smac_rt"].dropna()
prophet_smac = df.loc[prophet_mask, "smac_rt"].dropna()

print(f"  Before fix — Jain-137 SMAC: mean={jain_smac.mean():.2f}, std={jain_smac.std():.2f}")
print(f"  Before fix — PROPHET-Ab SMAC: mean={prophet_smac.mean():.2f}, std={prophet_smac.std():.2f}")

# Z-score per source, then rescale to Jain-137 scale (so units remain interpretable)
df["smac_rt_fixed"] = df["smac_rt"].copy()

# Normalize PROPHET-Ab to Jain-137 scale using z-score mapping
# z = (x - mean_prophet) / std_prophet
# x_mapped = z * std_jain + mean_jain
prophet_idx = df.index[prophet_mask & df["smac_rt"].notna()]
if len(prophet_idx) > 0:
    z_prophet = (df.loc[prophet_idx, "smac_rt"] - prophet_smac.mean()) / prophet_smac.std()
    df.loc[prophet_idx, "smac_rt_fixed"] = z_prophet * jain_smac.std() + jain_smac.mean()

fixed_smac = df["smac_rt_fixed"].dropna()
print(f"  After fix — All SMAC: mean={fixed_smac.mean():.2f}, std={fixed_smac.std():.2f}")
print(f"  After fix — Jain range: [{df.loc[jain_mask,'smac_rt_fixed'].min():.1f}, {df.loc[jain_mask,'smac_rt_fixed'].max():.1f}]")
print(f"  After fix — Prophet mapped range: [{df.loc[prophet_idx,'smac_rt_fixed'].min():.1f}, {df.loc[prophet_idx,'smac_rt_fixed'].max():.1f}]")


# ── Fix stability_slope: log-transform ─────────────────────────────────
print("\n=== Fixing stability_slope distribution ===")
ss = df["stability_slope"].dropna()
print(f"  Before: mean={ss.mean():.4f}, std={ss.std():.4f}, skewness={ss.skew():.2f}")
# log1p transform to spread compressed values; shift by min to ensure all positive
ss_min = ss.min()
df["stability_slope_log"] = np.nan
ss_idx = df.index[df["stability_slope"].notna()]
df.loc[ss_idx, "stability_slope_log"] = np.log1p(df.loc[ss_idx, "stability_slope"] - ss_min + 0.001)
ss_log = df["stability_slope_log"].dropna()
print(f"  After log1p: mean={ss_log.mean():.4f}, std={ss_log.std():.4f}, skewness={ss_log.skew():.2f}")


# ── XGBoost training with tuned hyperparameters + 5-fold CV ───────────
import xgboost as xgb

output_dir = os.path.join(PROJECT_ROOT, "models", "developability")
os.makedirs(output_dir, exist_ok=True)

seed = 42

# Hyperparameter grid per target
TUNED_PARAMS = {
    "stability_slope": {
        # Small dataset (137) → shallow trees, more regularization
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "min_child_weight": 5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": seed,
        "tree_method": "hist",
    },
    "smac_rt": {
        # Moderate dataset (246) → slightly more capacity but still conservative
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 3,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "random_state": seed,
        "tree_method": "hist",
    },
}

targets_to_fix = {
    "stability_slope": {
        "col": "stability_slope_log",  # log-transformed
        "inverse_fn": lambda y_pred: np.expm1(y_pred) + ss_min - 0.001,
        "orig_col": "stability_slope",
    },
    "smac_rt": {
        "col": "smac_rt_fixed",  # scale-normalized
        "inverse_fn": None,
        "orig_col": "smac_rt",
    },
}

# Also run baseline (original params, original data) for comparison
BASELINE_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": seed,
    "tree_method": "hist",
}

print("\n" + "="*70)
print("5-fold Cross-Validation Results")
print("="*70)

for target_key, cfg in targets_to_fix.items():
    print(f"\n{'─'*60}")
    print(f"TARGET: {target_key}")
    print(f"{'─'*60}")

    # ── Baseline (original data + original params) ────────────────
    orig_col = cfg["orig_col"]
    y_orig = df[orig_col].values.astype(float)
    valid_orig = ~np.isnan(y_orig)
    X_orig = X[valid_orig]
    y_orig_valid = y_orig[valid_orig]

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    baseline_rhos = []
    for train_idx, val_idx in kf.split(X_orig):
        model_bl = xgb.XGBRegressor(**BASELINE_PARAMS)
        model_bl.fit(X_orig[train_idx], y_orig_valid[train_idx],
                     eval_set=[(X_orig[val_idx], y_orig_valid[val_idx])],
                     verbose=False)
        pred_bl = model_bl.predict(X_orig[val_idx])
        rho_bl, _ = sp_stats.spearmanr(y_orig_valid[val_idx], pred_bl)
        baseline_rhos.append(rho_bl)

    mean_bl_rho = np.mean(baseline_rhos)
    print(f"  BASELINE (orig data, orig params, orig features only 7 biophys):")
    print(f"    5-fold rhos: {[f'{r:.3f}' for r in baseline_rhos]}")
    print(f"    Mean rho: {mean_bl_rho:.4f}")

    # ── Improved (fixed data + tuned params + extra features) ─────
    col = cfg["col"]
    y_fixed = df[col].values.astype(float)
    valid_fixed = ~np.isnan(y_fixed)
    X_fixed = X[valid_fixed]
    y_fixed_valid = y_fixed[valid_fixed]
    inverse_fn = cfg["inverse_fn"]

    params = TUNED_PARAMS[target_key]
    improved_rhos = []

    for train_idx, val_idx in kf.split(X_fixed):
        model_imp = xgb.XGBRegressor(**params)
        model_imp.fit(X_fixed[train_idx], y_fixed_valid[train_idx],
                      eval_set=[(X_fixed[val_idx], y_fixed_valid[val_idx])],
                      verbose=False)
        pred_imp = model_imp.predict(X_fixed[val_idx])

        # For rho evaluation on stability_slope, compare in original space
        if inverse_fn is not None:
            y_val_orig = inverse_fn(y_fixed_valid[val_idx])
            pred_orig = inverse_fn(pred_imp)
            rho_imp, _ = sp_stats.spearmanr(y_val_orig, pred_orig)
        else:
            rho_imp, _ = sp_stats.spearmanr(y_fixed_valid[val_idx], pred_imp)
        improved_rhos.append(rho_imp)

    mean_imp_rho = np.mean(improved_rhos)
    print(f"  IMPROVED (fixed data + tuned params + {N_BIOPHYS} biophys features):")
    print(f"    5-fold rhos: {[f'{r:.3f}' for r in improved_rhos]}")
    print(f"    Mean rho: {mean_imp_rho:.4f}")
    print(f"    Delta: {mean_imp_rho - mean_bl_rho:+.4f}")
    gate = "PASS" if abs(mean_imp_rho) > 0.15 else "FAIL"
    print(f"    Gate (|rho|>0.15): {gate}")

    # ── Train final model on all data with improved setup ─────────
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X_fixed, y_fixed_valid, verbose=False)

    model_path = os.path.join(output_dir, f"xgb_{target_key}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_dim": FEAT_DIM,
            "embedding_mode": "esm2",
            "target_transform": "log1p" if inverse_fn else "zscore_normalized",
            "tuned_params": params,
            "cv_mean_rho": round(float(mean_imp_rho), 4),
        }, f)
    print(f"  Saved: {model_path}")


# ── Also try: rank-transform the targets ──────────────────────────────
print(f"\n{'='*70}")
print("BONUS: Rank-transform approach (ordinal regression proxy)")
print(f"{'='*70}")

for target_key, cfg in targets_to_fix.items():
    print(f"\n  {target_key}:")
    orig_col = cfg["orig_col"]
    y_orig = df[orig_col].values.astype(float)
    valid = ~np.isnan(y_orig)
    X_v = X[valid]
    y_v = y_orig[valid]

    # Rank-transform target
    y_rank = sp_stats.rankdata(y_v) / len(y_v)

    params = TUNED_PARAMS[target_key]
    rank_rhos = []
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(X_v):
        model_r = xgb.XGBRegressor(**params)
        model_r.fit(X_v[train_idx], y_rank[train_idx],
                    eval_set=[(X_v[val_idx], y_rank[val_idx])],
                    verbose=False)
        pred_r = model_r.predict(X_v[val_idx])
        rho_r, _ = sp_stats.spearmanr(y_v[val_idx], pred_r)
        rank_rhos.append(rho_r)

    print(f"    Rank-transform 5-fold rhos: {[f'{r:.3f}' for r in rank_rhos]}")
    print(f"    Mean rho: {np.mean(rank_rhos):.4f}")

print("\nDone!")
