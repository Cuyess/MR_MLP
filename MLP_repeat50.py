import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import optuna
import sys
import os

# 1. Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ==========================================
# Model definition
# ==========================================
def create_mlp(input_dim, hidden_layer_size=64, dropout_rate=0.4, learning_rate=1e-3, l2_reg=1e-4):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name='input_layer'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
    return model

# ==========================================
# Helper metric functions
# ==========================================
def compute_binary_metrics(y_true, y_pred):
    # y_true and y_pred are both 0/1 arrays
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    youden = recall + spec - 1
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return {
        'Accuracy': acc, 'Sensitivity': recall, 'Specificity': spec,
        'Precision': prec, 'Youden': youden, 'F1': f1
    }

def multiclass_macro_metrics(y_true_int, y_pred_int, class_indices):
    metrics_per_class = {c: compute_binary_metrics((y_true_int == c).astype(int), (y_pred_int == c).astype(int)) for c in class_indices}
    avg = {}
    for key in ['Accuracy','Sensitivity','Specificity','Precision','Youden','F1']:
        avg[key] = np.mean([metrics_per_class[c][key] for c in class_indices])
    return avg

def format_mean_std(arr, pct=True):
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if pct:
        return f"{mean*100:.1f}% ± {std*100:.1f}%"
    else:
        return f"{mean:.3f} ± {std:.3f}"

def safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2: return float('nan')
        return float(roc_auc_score(y_true, y_score))
    except:
        return float('nan')

# ==========================================
# Data loading and preprocessing (with label-mapping fix)
# ==========================================
def load_and_prep_data():
    print("Loading data...")
    # Handle potential encoding issues
    try:
        df = pd.read_csv("fdr.csv", index_col=0)
    except UnicodeDecodeError:
        df = pd.read_csv("fdr.csv", index_col=0, encoding='gbk')
        
    df.index = df['编号']
    y_df, X = df['Group'], df.iloc[:, 2:]

    # [Key] Label mapping: 1->TRD, 2->nTRD, 3->HC
    label_mapping = {
        1: 'TRD', 2: 'nTRD', 3: 'HC',
        '1': 'TRD', '2': 'nTRD', '3': 'HC'
    }
    y_df = y_df.map(label_mapping)
    
    if y_df.isna().any():
        print("Error: Unrecognized labels found in Group column (expected 1, 2, 3)")
        sys.exit(1)
    
    if 'Gender' in X.columns:
        gender = pd.get_dummies(X['Gender'], prefix='Gender')
        X = X.drop(columns=['Gender'])
    else:
        gender = None

    y_onehot = pd.get_dummies(y_df, prefix='Group')
    class_names = [c.split('_',1)[1] for c in y_onehot.columns]
    print(f"Detected Classes: {class_names}")

    if os.path.exists("external_test.csv"):
        external_test_file = "external_test.csv"
    elif os.path.exists("external test.csv"):
        external_test_file = "external test.csv"
    else:
        raise FileNotFoundError("Missing external test index file. Expected 'external_test.csv' or 'external test.csv'.")
    external_test_indices = pd.read_csv(external_test_file, header=None).values.flatten()
    external_test_mask = X.index.isin(external_test_indices)
    
    X_external_test = X.loc[external_test_mask]
    y_external_test = y_onehot.loc[external_test_mask]
    X_internal = X.loc[~external_test_mask]
    y_internal = y_onehot.loc[~external_test_mask]

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X_internal, y_internal, test_size=0.2, random_state=123)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_test_scaled = pd.DataFrame(scaler.transform(X_val_test), columns=X_val_test.columns, index=X_val_test.index)
    X_external_test_scaled = pd.DataFrame(scaler.transform(X_external_test), columns=X_external_test.columns, index=X_external_test.index)

    if gender is not None:
        gender_train = gender.loc[X_train.index]
        gender_val_test = gender.loc[X_val_test.index]
        gender_external_test = gender.loc[X_external_test.index]
        X_train_scaled = pd.concat([gender_train, X_train_scaled], axis=1)
        X_val_test_scaled = pd.concat([gender_val_test, X_val_test_scaled], axis=1)
        X_external_test_scaled = pd.concat([gender_external_test, X_external_test_scaled], axis=1)

    return (X_train_scaled, y_train, 
            X_val_test_scaled, y_val_test, 
            X_external_test_scaled, y_external_test, 
            class_names)

# ==========================================
# Optuna optimization function
# ==========================================
def run_optuna_search(X_train, y_train, n_trials=50):
    print(f"\n[Optuna] Starting hyperparameter search with {n_trials} trials...")
    
    # Apply SMOTE in advance to speed up Optuna search (official training still applies it in-loop)
    sm_opt = SMOTE(random_state=123)
    X_res, y_res = sm_opt.fit_resample(X_train, np.argmax(y_train.values, axis=1))
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res_series = pd.Series(y_res)
    
    # Get number of classes
    n_classes = y_train.shape[1]

    def objective(trial):
        hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 100)
        dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True)
        epochs = trial.suggest_int('epochs', 20, 100)
        batch_size = trial.suggest_int('batch_size', 16, 128)
        
        # Use 3-fold CV for quick evaluation
        kf = KFold(n_splits=3, shuffle=True, random_state=123)
        scores = []
        
        for tr_idx, val_idx in kf.split(X_res, y_res_series):
            X_tr, X_val = X_res.iloc[tr_idx], X_res.iloc[val_idx]
            y_tr, y_val = y_res_series.iloc[tr_idx], y_res_series.iloc[val_idx]
            y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes=n_classes)
            y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

            model = create_mlp(X_tr.shape[1], hidden_layer_size, dropout_rate, learning_rate, l2_reg)
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            model.fit(X_tr, y_tr_oh, epochs=epochs, batch_size=batch_size, 
                      validation_data=(X_val, y_val_oh), callbacks=[es], verbose=0)
            
            # Use Macro AUC as the optimization objective
            preds = model.predict(X_val, verbose=0)
            try:
                # Compute Macro AUC directly
                aucs = []
                for c in range(n_classes):
                    aucs.append(roc_auc_score((y_val == c).astype(int), preds[:, c]))
                scores.append(np.mean(aucs))
            except:
                scores.append(0.5) # Failure penalty
        
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("[Optuna] Best params found:", study.best_params)
    return study.best_params

# ==========================================
# Main execution logic
# ==========================================
def main(n_repeats=50, do_optuna=True):
    # 1. Load data
    data = load_and_prep_data()
    X_train_s, y_train, X_int_test_s, y_int_test, X_ext_test_s, y_ext_test, class_names = data
    
    # 2. Determine hyperparameters
    if do_optuna:
        best_params = run_optuna_search(X_train_s, y_train, n_trials=50) # Number of trials can be adjusted here
    else:
        # If Optuna is disabled, use default parameters
        best_params = {
            'hidden_layer_size': 64, 'dropout_rate': 0.4, 'learning_rate': 1e-3, 
            'l2_reg': 1e-4, 'epochs': 50, 'batch_size': 32
        }
    
    # Fill in any missing parameters
    defaults = {'patience': 5, 'epochs': 50, 'batch_size': 32}
    for k, v in defaults.items():
        if k not in best_params: best_params[k] = v
            
    print(f"\n[Execution] Starting {n_repeats} runs using params: {best_params}\n")

    # 3. Initialize storage containers
    results_internal = {k: [] for k in ['Macro Average', 'TRD vs nTRD+HC', 'nTRD vs TRD+HC', 'HC vs TRD+nTRD', 'TRD vs nTRD', 'TRD vs HC', 'nTRD vs HC']}
    results_external = {k: [] for k in results_internal.keys()}
    
    cm_int_history = []
    cm_ext_history = []
    
    # ROC data container for confidence-interval plotting
    mean_fpr = np.linspace(0, 1, 100) # Unified x-axis
    roc_data = {'Internal': {}, 'External': {}}
    for ds in ['Internal', 'External']:
        for curve_type in ['TRD vs Others', 'nTRD vs Others', 'HC vs Others', 'Macro Average', 'TRD vs nTRD', 'TRD vs HC']:
            roc_data[ds][curve_type] = []

    # Prepare class indices
    try:
        trd_idx = class_names.index('TRD')
        ntrd_idx = class_names.index('nTRD')
        hc_idx = class_names.index('HC')
    except:
        trd_idx, ntrd_idx, hc_idx = 0, 1, 2
        
    class_indices = list(range(len(class_names)))
    y_int_test_int = np.argmax(y_int_test.values, axis=1)
    y_ext_test_int = np.argmax(y_ext_test.values, axis=1)

    # 4. Repeat 50 runs
    for i in range(1, n_repeats + 1):
        seed = 1000 + i
        
        # SMOTE
        sm = SMOTE(random_state=seed)
        X_res, y_res = sm.fit_resample(X_train_s, np.argmax(y_train.values, axis=1))
        X_res = pd.DataFrame(X_res, columns=X_train_s.columns)
        y_res_series = pd.Series(y_res)
        
        # 5-Fold CV Ensemble
        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        preds_val_agg = np.zeros((X_int_test_s.shape[0], 3))
        preds_ext_agg = np.zeros((X_ext_test_s.shape[0], 3))
        
        for train_idx, val_idx in kf.split(X_res, y_res_series):
            X_fold_tr = X_res.iloc[train_idx]
            y_fold_tr = tf.keras.utils.to_categorical(y_res_series.iloc[train_idx], num_classes=3)
            X_fold_val = X_res.iloc[val_idx]
            y_fold_val = tf.keras.utils.to_categorical(y_res_series.iloc[val_idx], num_classes=3)
            
            model = create_mlp(X_fold_tr.shape[1], 
                               best_params['hidden_layer_size'], 
                               best_params['dropout_rate'], 
                               best_params['learning_rate'], 
                               best_params['l2_reg'])
            
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=best_params['patience'], restore_best_weights=True)
            model.fit(X_fold_tr, y_fold_tr, epochs=best_params['epochs'], batch_size=best_params['batch_size'], 
                      validation_data=(X_fold_val, y_fold_val), callbacks=[es], verbose=0)
            
            preds_val_agg += model.predict(X_int_test_s, verbose=0)
            preds_ext_agg += model.predict(X_ext_test_s, verbose=0)
            
        preds_val_avg = preds_val_agg / 5.0
        preds_ext_avg = preds_ext_agg / 5.0
        
        y_int_pred = np.argmax(preds_val_avg, axis=1)
        y_ext_pred = np.argmax(preds_ext_avg, axis=1)
        
        # --- (A) Record confusion matrices ---
        cm_int_history.append(confusion_matrix(y_int_test_int, y_int_pred, labels=class_indices))
        cm_ext_history.append(confusion_matrix(y_ext_test_int, y_ext_pred, labels=class_indices))
        
        # --- (B) Record metrics ---
        def process_metrics(y_true, y_pred, probs, res_dict):
            # Macro
            res_dict['Macro Average'].append(multiclass_macro_metrics(y_true, y_pred, class_indices))
            res_dict['Macro Average'][-1]['AUC'] = safe_auc(tf.keras.utils.to_categorical(y_true, 3), probs)
            
            # Binary Comparisons
            comparisons = [
                ('TRD vs nTRD+HC', trd_idx, None),
                ('nTRD vs TRD+HC', ntrd_idx, None),
                ('HC vs TRD+nTRD', hc_idx, None)
            ]
            for name, cls_idx, _ in comparisons:
                y_bin = (y_true == cls_idx).astype(int)
                y_p_bin = (y_pred == cls_idx).astype(int)
                m = compute_binary_metrics(y_bin, y_p_bin)
                m['AUC'] = safe_auc(y_bin, probs[:, cls_idx])
                res_dict[name].append(m)
                
            # Pairwise
            pairs = [
                ('TRD vs nTRD', trd_idx, ntrd_idx),
                ('TRD vs HC', trd_idx, hc_idx),
                ('nTRD vs HC', ntrd_idx, hc_idx)
            ]
            for name, c1, c2 in pairs:
                mask = np.isin(y_true, [c1, c2])
                if mask.sum() > 0:
                    y_sub = (y_true[mask] == c1).astype(int)
                    y_p_sub = (y_pred[mask] == c1).astype(int)
                    m = compute_binary_metrics(y_sub, y_p_sub)
                    m['AUC'] = safe_auc(y_sub, probs[mask, c1])
                    res_dict[name].append(m)

        process_metrics(y_int_test_int, y_int_pred, preds_val_avg, results_internal)
        process_metrics(y_ext_test_int, y_ext_pred, preds_ext_avg, results_external)

        # --- (C) [Key] Collect ROC curve shapes ---
        def collect_roc(y_true, y_probs, dataset_key):
            # 1. One-vs-Rest
            for cls_idx, cls_name in zip([trd_idx, ntrd_idx, hc_idx], ['TRD', 'nTRD', 'HC']):
                y_bin = (y_true == cls_idx).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, y_probs[:, cls_idx])
                # Linear interpolation to the unified x-axis
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                roc_data[dataset_key][f'{cls_name} vs Others'].append(tpr_interp)
            
            # 2. Macro (Average of TPRs)
            tprs_list = []
            for cls_idx in class_indices:
                y_bin = (y_true == cls_idx).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, y_probs[:, cls_idx])
                tprs_list.append(np.interp(mean_fpr, fpr, tpr))
            macro_tpr = np.mean(tprs_list, axis=0)
            macro_tpr[0] = 0.0
            roc_data[dataset_key]['Macro Average'].append(macro_tpr)
            
            # 3. Pairwise
            for cls_name, c1, c2 in [('TRD vs nTRD', trd_idx, ntrd_idx), ('TRD vs HC', trd_idx, hc_idx)]:
                mask = np.isin(y_true, [c1, c2])
                if mask.sum() > 0:
                    y_bin = (y_true[mask] == c1).astype(int)
                    fpr, tpr, _ = roc_curve(y_bin, y_probs[mask, c1])
                    tpr_interp = np.interp(mean_fpr, fpr, tpr)
                    tpr_interp[0] = 0.0
                    roc_data[dataset_key][cls_name].append(tpr_interp)

        collect_roc(y_int_test_int, preds_val_avg, 'Internal')
        collect_roc(y_ext_test_int, preds_ext_avg, 'External')
        
        print(f"Run {i}/{n_repeats} completed.")

    # 5. Save all output files
    print("\nSaving final outputs...")
    
    # 1. Save ROC data (for plotting)
    np.save('roc_curves_data.npy', roc_data)
    print("Saved: roc_curves_data.npy")
    
    # 2. Save metric summary (Table 3)
    rows = []
    for d_name, d_dict in [('Internal Test', results_internal), ('External Test', results_external)]:
        for cat_name, metrics_list in d_dict.items():
            row = {'Dataset': d_name, 'Category': cat_name}
            for k in ['Accuracy','Sensitivity','Specificity','Precision','Youden','F1','AUC']:
                vals = [m.get(k, np.nan) for m in metrics_list]
                vals = np.array(vals, dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    pct = k not in ['Youden', 'AUC']
                    row[k if k!='Sensitivity' else 'Sensitivity/Recall'] = format_mean_std(vals, pct=pct)
                    if k == 'F1': row['F1 Score'] = row.pop('F1')
                    if k == 'Youden': row['Youden Index'] = row.pop('Youden')
                else:
                    row[k] = "N/A"
            rows.append(row)
    pd.DataFrame(rows).to_csv('model1_repeat50_summary.csv', index=False)
    print("Saved: model1_repeat50_summary.csv")
    
    # 3. Save confusion matrix statistics (Figure 2)
    cm_rows = []
    for d_name, stack in [('Internal Test', np.array(cm_int_history)), ('External Test', np.array(cm_ext_history))]:
        mean_mat = np.mean(stack, axis=0)
        std_mat = np.std(stack, axis=0)
        for r in range(3):
            for c in range(3):
                cm_rows.append({
                    'Dataset': d_name,
                    'True Label': class_names[r],
                    'Predicted Label': class_names[c],
                    'Mean': mean_mat[r,c],
                    'Std': std_mat[r,c],
                    'Text': f"{mean_mat[r,c]:.1f}\n(±{std_mat[r,c]:.1f})"
                })
    pd.DataFrame(cm_rows).to_csv('confusion_matrix_stats.csv', index=False)
    print("Saved: confusion_matrix_stats.csv")
    
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    # do_optuna=True runs Optuna search first
    # n_repeats=50 runs 50 validation repeats
    main(n_repeats=50, do_optuna=True)
