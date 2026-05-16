# Data Directory

## Included Data (tracked in git)

| File | Size | Description |
|------|------|-------------|
| `Jain137_Cleaned_Training_Data.csv` | 139 rows | Primary labeled dataset (13 biophysical targets) |
| `unified_training_data.csv` | 954 rows | Merged training data for unified multitask model |
| `merged_xgb_training.csv` | 370 rows | Merged XGBoost training data |
| `TheraSAbDab_SeqStruc_OnlineDownload.csv` | 1,133 rows | Therapeutic antibody sequences + formats |
| `reference/benchmark_sequences.json` | 12 molecules | Full HC+LC sequences from UniProt/KEGG/PDB |
| `reference/cief_reference_profiles.json` | 20 profiles | Published cIEF charge variant data |
| `reference/molecule_benchmarks.csv` | 12 molecules | Multi-type benchmark with published values |
| `flab/*.csv` | Various | FLAb assay data (Tm, SEC, HIC, AC-SINS, etc.) |

## Excluded Data (download separately)

These files are too large for git but are needed for full functionality.

### Download Instructions

**CoV-AbDab** (~8MB each):
```bash
# Download from: https://opig.stats.ox.ac.uk/webapps/covabdab/
# Save as: data/CoV-AbDab.csv
```

**SAbDab Summary** (~8MB):
```bash
# Download from: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/
# Save as: data/sabdab_summary_all.tsv
```

**ESM-2 Embedding Cache** (~3.5MB, regeneratable):
```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/build_esm2_cache.py
# Generates: data/esm2_embeddings_cache.pt
```

**Classifier Training Data** (~7MB, regeneratable):
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from training.data_harmonizer import harmonize
harmonize(data_dir='data', output_path='data/training/classifier_data.csv')
"
```

**DRAMP / ThPD / AMP databases** (optional, for peptide classification):
```bash
# DRAMP: https://dramp.cpu-bioinfor.org/downloads/
# ThPD: http://crdd.osdd.net/raghava/thpdb/
# Save as: data/DRAMP_general.xlsx, data/ThPD.xlsx
```

## Regenerating All Data

To regenerate all derived data from scratch:
```bash
cd /path/to/ProtePilot

# 1. Unified training data
python3 scripts/build_integrated_training_data.py

# 2. ESM-2 embeddings
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/build_esm2_cache.py

# 3. Classifier data
python3 -c "import sys; sys.path.insert(0, 'src'); from training.data_harmonizer import harmonize; harmonize(data_dir='data', output_path='data/training/classifier_data.csv')"
```
