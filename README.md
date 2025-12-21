# MR_MLP
# MR_MLP: Brain Analysis with MLP & SHAP Interpretability

This repository contains a machine learning pipeline for analyzing brain imaging data using Multilayer Perceptrons (MLP). It includes a hybrid workflow using **R** for significant feature selection and **Python** for model training, performance evaluation, and SHAP-based interpretability visualization.

## 📂 Project Structure

Here is an overview of the files in this repository and their specific roles:

| File Name | Description |
| :--- | :--- |
| **`select_sig_idx.R`** | **Preprocessing & Feature Selection.** An R script used to identify significant indices or features (based on inner dataset) from the raw dataset before feeding them into the model. |
| **`MLP.py`** | **Model Implementation.** Contains the architecture of the Multilayer Perceptron (MLP) and the data processing pipeline required to train the model. |
| **`performance_evaluation.py`** | **Metrics & Validation.** Scripts to evaluate the trained model's performance (e.g., Accuracy, ROC-AUC, Sensitivity, Specificity). |
| **`calc_SHAP.py`** | **Model Interpretability.** Loads the trained model and data to calculate SHAP (SHapley Additive exPlanations) values, identifying which features contribute most to the predictions. |
| **`brainmap.py`** | **Visualization.** Visualizes the calculated SHAP values or feature importance onto brain maps/atlases for interpretation. |

