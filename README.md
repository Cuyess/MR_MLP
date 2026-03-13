# MR_MLP

An MLP-based neuroimaging three-classification pipeline (TRD / nTRD / HC), including:

- **R-side feature selection** (ANCOVA + FDR)
- **Python-side modeling and validation** (Optuna + SMOTE + 50 repeats)
- **Interpretability analysis** (SHAP)
- **Result visualization** (ROC, confusion matrix, brain maps)

---

## 1. Project Files

| File | Purpose |
| :--- | :--- |
| `select_sig_idx.R` | Performs significance filtering from raw features and outputs `fdr.csv` (input for Python training). |
| `MLP.py` | MLP training and basic evaluation script (simplified workflow). |
| `MLP_repeat50.py` | Main workflow: split data, standardize, SMOTE, Optuna tuning, 5-fold ensemble, repeat 50 times, and summarize outputs. |
| `plot_MLP.py` | Loads model outputs and plots ROC curves plus confusion matrices. |
| `calc_SHAP.py` | Loads a trained model and data, computes SHAP, and exports `real_shap.csv`. |
| `brainmap.py` | Builds brain-region visualizations from `real_shap.csv` and atlas metadata. |

---

## 2. Environment

This repository does not include `requirements.txt`. Install dependencies manually based on script imports.

### Python dependencies

```bash
pip install pandas numpy tensorflow scikit-learn optuna imbalanced-learn shap matplotlib seaborn nilearn openpyxl
```

### R dependencies

```r
install.packages(c("readxl", "multcomp"))
```

---

## 3. Required Data and Files

At minimum, prepare the following inputs (filenames should match script expectations):

- `external_test.csv` (recommended) or `external test.csv` (legacy-compatible): external test sample IDs  
  (underscore naming is recommended to avoid shell/path escaping issues on different platforms)
- Raw tables for the R step (for example: `ALFF.xlsx`, `ALFF_sc_fc_coordination.xlsx`, `cov.xlsx`)
- SHAP and brain-map steps additionally require:
  - `final_model.keras`
  - `X_train.npy`, `X_outer_test.npy`, `y_outer_test.npy`, `lable.npy`
  - `BrainnetomeAtlas_BNA_subregions.xlsx`
  - `brainmap_python/ALFF.nii`, `brainmap_python/SFC.nii`

> Note: `final_model.keras` and the `.npy` files above must be produced by your existing training workflow or prepared manually. Current scripts read them directly.

> Note: `select_sig_idx.R` still contains author-local absolute paths (`setwd(...)` and some read paths). Update them to your own local paths or relative paths before running.

---

## 4. Recommended Execution Order

### Quickstart (most common)

```bash
Rscript select_sig_idx.R
python MLP_repeat50.py
python plot_MLP.py
```

> These 3 steps cover the core workflow: feature filtering + model evaluation + Figure 2 plotting.

### Step 1) R feature selection

```bash
Rscript select_sig_idx.R
```

Output: `fdr.csv`

### Step 2) Python training and validation (main workflow)

```bash
python MLP_repeat50.py
```

Main outputs:

- `roc_curves_data.npy`
- `model1_repeat50_summary.csv`
- `confusion_matrix_stats.csv`

> For a simpler workflow, run `python MLP.py`.

### Step 3) Result figures (ROC / confusion matrix)

```bash
python plot_MLP.py
```

Main outputs:

- `Figure2_MeanOnly_CustomColor.pdf`
- `Figure2_MeanOnly_CustomColor.png`

### Step 4) SHAP calculation

```bash
python calc_SHAP.py
```

Main outputs:

- `shap.npy`
- `real_shap.csv`

### Step 5) Brain-map visualization

```bash
python brainmap.py
```

Main output directory:

- `visualization_results/brain_visualization/`

---

## 5. Notes

- This repository currently has no automated tests, lints, or CI workflow configured.
- Scripts were written in a paper-reproduction style. Some input filenames and paths are fixed. For new datasets, first standardize paths and column names.

---

## 6. FAQ

**Q1: `select_sig_idx.R` fails with path errors. Why?**  
A: The script includes author-local absolute paths. Replace them with your own local paths or relative paths.

**Q2: What should the external test index filename be?**  
A: `external_test.csv` is recommended. Current code also supports legacy `external test.csv`.

**Q3: Can I skip SHAP and only generate model performance figures?**  
A: Yes. Run only Steps 1–3 to obtain ROC and confusion matrix figures (`Figure2_MeanOnly_CustomColor.*`).

**Q4: Why can I see changes in the Pull Request, but not in the Code tab?**  
A: The **Code** tab usually shows the repository default branch (often `main`), while your PR changes are on a feature branch. Until merged, those edits may not appear in default-branch Code view.

Use one of these methods to view/edit updated files:

1. In GitHub UI, switch branch in the branch dropdown (top-left in Code view) to your PR branch.
2. In your local repo, checkout the PR branch:

```bash
git fetch origin
git checkout <your-pr-branch-name>
git pull
```

Then open and edit `/home/runner/work/MR_MLP/MR_MLP/README.md` (or any changed file), commit to the same branch, and push.
