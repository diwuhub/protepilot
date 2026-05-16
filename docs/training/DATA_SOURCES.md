# Training Data Sources — ProtePilot Platform

## Currently Integrated (auto-loaded)

| Source | File | Entries | Classes | Status |
|--------|------|---------|---------|--------|
| **Jain-137** | `data/Jain137_Cleaned_Training_Data.csv` | 137 | canonical_mab | ✅ Auto-loaded |
| **TheraSAbDab** | `data/TheraSAbDab_SeqStruc_OnlineDownload.csv` | 1,133 | 7 classes | ✅ Auto-loaded |
| **CoV-AbDab** | `data/CoV-AbDab.csv` | 12,346 | canonical_mab, single_domain | ✅ Integrated 2026-03-19 |
| **DRAMP 4.0** | `data/DRAMP_general.xlsx` | 5,000 (sampled from 11,612) | peptide | ✅ Integrated 2026-03-19 |
| **ThPD** | `data/ThPD.xlsx` | 5,000 (sampled from 58,583) | peptide | ✅ Integrated 2026-03-19 |
| **Synthetic** | Generated at runtime | 65 | peptide, nanobody, scaffold | ✅ Auto-generated |

**Total harmonized dataset: 23,668 rows, 8 classes** (as of 2026-03-19)

## Ready to Integrate (download → place → retrain)

### 1. SAbDab — Structural Antibody Database

~18,700 antibody Fv structures from PDB. Best for: CDR feature extraction, structural liability prediction.

**Download:**
1. Go to https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/
2. Click **"Download"** → select TSV summary
3. Save as `data/SAbDab_summary.tsv`

**What it adds:** VH/VL sequences with chain subgroup annotations, antigen info, resolution, species.

**Expected impact:** Classifier gets structural diversity. Can extract CDR loop features (CDRH3 length, hydrophobicity) for better single_domain vs canonical_mab discrimination.

**Note:** Standard download is summary-only (no sequences). Need the full sequence export for the adapter to load entries.

### 2. IMGT/mAb-DB — International Immunogenetics Monoclonal Antibody Database

~1,855 therapeutic antibodies with INN names, format classification, target info.

**Download:**
1. Go to https://www.imgt.org/mAb-DB/index
2. Use query interface → search all entries
3. Export results → save as `data/IMGT_mAbDB.csv`
4. Ensure columns include: INN (or Name), Format, VH, VL

**What it adds:** Curated therapeutic mAb metadata with authoritative format labels (IgG1, bispecific, ADC, Fc-fusion, etc.)

**Expected impact:** Higher-quality labels for bispecific/ADC/Fc-fusion — currently these are under-represented.

## How to Use Downloaded Data

After downloading any of these files into `data/`, simply retrain:

### From command line:
```bash
python -m src.training.data_harmonizer --output data/training/classifier_data.csv
python -m src.training.classifier_trainer
python -m src.training.ood_trainer
python -m src.training.benchmark_evaluator
```

### From Streamlit UI:
1. Open the app → **AI Training Center** → **Molecule Classifier & OOD** tab
2. Click **"Train Molecule Classifier"**
3. Click **"Train OOD Detector"**

The harmonizer automatically detects which files exist in `data/` and includes them. No code changes needed.

## Future Sources (not yet integrated)

| Source | URL | Entries | What it adds |
|--------|-----|---------|-------------|
| **DrugBank** | drugbank.ca | ~2,500 biologics | PK parameters, ADA reports, indications |
| **KEGG DRUG** | genome.jp/kegg/drug/ | ~1,000 biologics | ATC codes, target pathways |
| **UniProt** (biologics) | uniprot.org | ~5,000 | Function annotations, subcellular localization |
| **PDB** (antibodies) | rcsb.org | ~4,000 Ab | 3D structure, CDR conformations |
| **CoV-UniBind** | (preprint) | ~75,000 | Binding affinity data for SARS-CoV-2 antibodies |

These would require additional adapter code in `data_harmonizer.py`. Open an issue or ask to implement.

找到了多个高质量的 peptide 数据源。以下是按实用性排序的推荐：

---

## Peptide 训练数据来源

### 最推荐（数据量大、下载方便、标签清楚）

| 数据库 | 条目数 | 下载格式 | URL | 特点 |
|--------|--------|---------|-----|------|
| **Comprehensive Therapeutic Peptide Dataset (2025)** | 58,583 | CSV (FigShare) | [FigShare](https://doi.org/10.6084/m9.figshare.28691885) | 47 功能分类，有结构注释，2025年1月发表，质量最高 |
| **DRAMP 4.0** | 30,260 | FASTA / xlsx | [dramp.cpu-bioinfor.org](http://dramp.cpu-bioinfor.org/) | 抗菌肽为主，有临床阶段标注，xlsx 可直接用 |
| **SATPdb** | 19,192 | FASTA | [satpdb/down.php](http://crdd.osdd.net/raghava/satpdb/down.php) | 按功能分10类（抗癌、抗菌、降压等），结构注释 |

### 次推荐（更小但更精确）

| 数据库 | 条目数 | 下载格式 | URL | 特点 |
|--------|--------|---------|-----|------|
| **THPdb2** | 85 peptides + 354 mAbs | Web interface | [webs.iiitd.edu.in/raghava/thpdb2](https://webs.iiitd.edu.in/raghava/thpdb2/) | FDA 批准的治疗性肽，数量少但标签权威 |
| **Peptipedia v2.0** | 100,000+ | Web search | [peptipedia.cl](https://peptipedia.cl/) | 最大的公开肽数据库，但需要筛选治疗性肽 |
| **APD3** (Antimicrobial Peptide Database) | 3,300+ | FASTA | [aps.unmc.edu](https://aps.unmc.edu/) | 天然抗菌肽，有活性注释 |

### 操作方法

**最快的方式：下载 FigShare 上的 Comprehensive Therapeutic Peptide Dataset**

1. 打开 https://doi.org/10.6084/m9.figshare.28691885
2. 下载 CSV 文件
3. 保存到 `data/` 文件夹（我会写适配器）

**DRAMP 4.0：**

1. 打开 http://dramp.cpu-bioinfor.org/
2. 点 Download → 选 xlsx 格式
3. 保存到 `data/DRAMP_general.xlsx`

你下载哪个告诉我，我立刻写适配器接入 harmonizer 并重新训练。当前 peptide 只有 30 条 synthetic，接入任何一个上面的库都会显著改善 peptide F1（现在 benchmark 上 GLP-1 被错分为 single_domain 就是因为 peptide 样本太少）。

Sources:
- [Comprehensive Therapeutic Peptide Dataset (2025)](https://www.nature.com/articles/s41597-025-05528-1)
- [DRAMP 4.0](https://academic.oup.com/nar/article/53/D1/D403/7889245)
- [SATPdb](https://academic.oup.com/nar/article/44/D1/D1119/2502618)
- [THPdb2](https://webs.iiitd.edu.in/raghava/thpdb2/)
- [Peptipedia v2.0](https://academic.oup.com/database/article/doi/10.1093/database/baae113/7887558)
- [APD3](https://aps.unmc.edu/)