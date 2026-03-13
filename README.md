# MR_MLP

基于 MLP 的脑影像三分类分析流程（TRD / nTRD / HC），包含：

- **R 侧特征筛选**（ANCOVA + FDR）
- **Python 侧建模与验证**（Optuna + SMOTE + 50 次重复）
- **可解释性分析**（SHAP）
- **结果可视化**（ROC、混淆矩阵、脑图）

---

## 1. 项目文件说明

| 文件 | 作用 |
| :--- | :--- |
| `select_sig_idx.R` | 从原始特征中做显著性筛选，输出 `fdr.csv`（后续 Python 训练输入）。 |
| `MLP.py` | MLP 训练与基础评估脚本（较简版流程）。 |
| `MLP_repeat50.py` | 主流程脚本：数据划分、标准化、SMOTE、Optuna 调参、5-fold 集成、重复 50 次并汇总结果。 |
| `plot_MLP.py` | 读取模型输出，绘制 ROC 曲线与混淆矩阵图。 |
| `calc_SHAP.py` | 加载训练好的模型与数据，计算 SHAP 并导出 `real_shap.csv`。 |
| `brainmap.py` | 基于 `real_shap.csv` 和脑图谱信息绘制脑区可视化。 |

---

## 2. 运行环境

本仓库未提供 `requirements.txt`，可按代码依赖手动安装。

### Python 依赖

```bash
pip install pandas numpy tensorflow scikit-learn optuna imbalanced-learn shap matplotlib seaborn nilearn openpyxl
```

### R 依赖

```r
install.packages(c("readxl", "multcomp"))
```

---

## 3. 数据与文件准备

按脚本逻辑，至少需要以下输入（文件名需与代码一致）：

- `external_test.csv`（推荐）或 `external test.csv`（兼容）：外部测试集样本编号  
  （推荐下划线命名，避免文件名空格在命令行或跨平台环境中带来的路径转义问题）
- R 步骤需要的原始表格（如 `ALFF.xlsx`、`ALFF_sc_fc_coordination.xlsx`、`cov.xlsx`）
- SHAP 与脑图步骤还需要：
  - `final_model.keras`
  - `X_train.npy`, `X_outer_test.npy`, `y_outer_test.npy`, `lable.npy`
  - `BrainnetomeAtlas_BNA_subregions.xlsx`
  - `brainmap_python/ALFF.nii`, `brainmap_python/SFC.nii`

> 注意：`select_sig_idx.R` 中包含作者本地绝对路径（`setwd(...)` 和部分读文件路径），使用前请改为你自己的路径或相对路径。

---

## 4. 推荐执行顺序

### 一键快速开始（最常用）

```bash
Rscript select_sig_idx.R
python MLP_repeat50.py
python plot_MLP.py
```

> 上面 3 步可先完成“特征筛选 + 训练评估 + Figure2 绘图”主流程。

### Step 1) R 特征筛选

```bash
Rscript select_sig_idx.R
```

输出：`fdr.csv`

### Step 2) Python 训练与验证（主流程）

```bash
python MLP_repeat50.py
```

主要输出：

- `roc_curves_data.npy`
- `model1_repeat50_summary.csv`
- `confusion_matrix_stats.csv`

> 如需简化版流程，可运行 `python MLP.py`。

### Step 3) 结果图（ROC / 混淆矩阵）

```bash
python plot_MLP.py
```

主要输出：

- `Figure2_MeanOnly_CustomColor.pdf`
- `Figure2_MeanOnly_CustomColor.png`

### Step 4) SHAP 计算

```bash
python calc_SHAP.py
```

主要输出：

- `shap.npy`
- `real_shap.csv`

### Step 5) 脑图可视化

```bash
python brainmap.py
```

主要输出目录：

- `visualization_results/brain_visualization/`

---

## 5. 说明

- 当前仓库未配置自动化测试、lint 或 CI 工作流。
- 脚本以“论文复现实验脚本”风格编写，部分输入文件名和路径是固定写法；如用于新数据，请先统一路径与字段名。

---

## 6. 常见问题（FAQ）

**Q1：运行 `select_sig_idx.R` 报路径错误？**  
A：脚本中写了作者本地绝对路径，请改为你本机路径或相对路径后再运行。

**Q2：外部测试索引文件应该叫什么名字？**  
A：推荐使用 `external_test.csv`。当前代码也兼容旧命名 `external test.csv`。

**Q3：只想先看模型性能图，不做 SHAP 可以吗？**  
A：可以。先执行 Step 1~3，即可得到 ROC 和混淆矩阵图（`Figure2_MeanOnly_CustomColor.*`）。
