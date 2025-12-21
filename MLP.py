import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import optuna
from imblearn.over_sampling import SMOTE

df = pd.read_csv("fdr.csv", index_col=0)
df.index = df['编号']
y, X = df['Group'], df.iloc[:, 2:]
y = pd.get_dummies(y, prefix='Group')
gender = pd.get_dummies(X['Gender'], prefix='Gender')
X = X.drop(columns=['Gender'])

external_test_indices = pd.read_csv("external test.csv", header=None).values.flatten()

external_test_mask = X.index.isin(external_test_indices)
X_external_test = X.loc[external_test_mask]
y_external_test = y.loc[external_test_mask]
gender_external_test = gender.loc[external_test_mask]
np.save("y_external_id.npy", y_external_test.index)

X_internal = X.loc[~external_test_mask]
y_internal = y.loc[~external_test_mask]
gender_internal = gender.loc[~external_test_mask]

# split
X_train, X_val_test, gender_train, gender_val_test, y_train, y_val_test = train_test_split(
    X_internal, gender_internal, y_internal, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_test_scaled = pd.DataFrame(scaler.transform(X_val_test), columns=X_val_test.columns, index=X_val_test.index)
X_external_test_scaled = pd.DataFrame(scaler.transform(X_external_test), columns=X_external_test.columns, index=X_external_test.index)

X_train_scaled = pd.concat([gender_train, X_train_scaled], axis=1)
X_val_test_scaled = pd.concat([gender_val_test, X_val_test_scaled], axis=1)
X_external_test_scaled = pd.concat([gender_external_test, X_external_test_scaled], axis=1)

y_train_int = np.argmax(y_train.values, axis=1)
y_val_test_int = np.argmax(y_val_test.values, axis=1)
y_external_test_int = np.argmax(y_external_test.values, axis=1)

# SMOTE
smote = SMOTE(random_state=123)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_int)
print(X_train_resampled.shape)

X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train_scaled.columns)
X_train_resampled = X_train_resampled.reset_index(drop=True)
y_train_resampled = pd.Series(y_train_resampled, name='Group')

def create_mlp(hidden_layer_size, dropout_rate, learning_rate, l2_reg):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],), name='input_layer'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(3, activation='softmax')  
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
    return model

def compute_composite_metric(y_true, y_pred_proba, weight_pairwise=0):
    binary_labels = y_true[:, 0].astype(int)
    y_pred_0 = y_pred_proba[:, 0]
    pairwise_auc = roc_auc_score(binary_labels, y_pred_0)
    global_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
    composite_metric = weight_pairwise * pairwise_auc + (1 - weight_pairwise) * global_auc
    return composite_metric, pairwise_auc, global_auc

def evaluate_mlp(trial):
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 100)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 10, 100)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    patience = trial.suggest_int('patience', 5, 20)

    model = create_mlp(hidden_layer_size=hidden_layer_size, dropout_rate=dropout_rate, learning_rate=learning_rate, l2_reg=l2_reg)

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    cv_scores = []
    for train_index, val_index in kf.split(X_train_resampled, y_train_resampled):  # 使用SMOTE增强后的数据
        X_train_fold, X_val_fold = X_train_resampled.iloc[train_index], X_train_resampled.iloc[val_index]
        y_train_fold, y_val_fold = y_train_resampled[train_index], y_train_resampled[val_index]

        # one hot
        y_train_fold_onehot = tf.keras.utils.to_categorical(y_train_fold, num_classes=3)
        y_val_fold_onehot = tf.keras.utils.to_categorical(y_val_fold, num_classes=3)

        # early stop
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        model.fit(
            X_train_fold, y_train_fold_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_fold, y_val_fold_onehot),
            callbacks=[early_stopping],
            verbose=0
        )

        y_pred_proba = model.predict(X_val_fold)
        _, _, global_auc = compute_composite_metric(y_val_fold_onehot, y_pred_proba)
        cv_scores.append(global_auc)

    avg_composite_metric = float(np.mean(cv_scores))
    return avg_composite_metric

def main():

    study = optuna.create_study(direction='maximize')
    study.optimize(evaluate_mlp, n_trials=20, n_jobs=-1)
    
    print("Best parameters found: ", study.best_params)
    print("Best Composite Metric: ", study.best_value)
    
    best_params = study.best_params
    final_model = create_mlp(
        hidden_layer_size=best_params['hidden_layer_size'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        l2_reg=best_params['l2_reg']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=best_params['patience'],
        restore_best_weights=True
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    for train_index, val_index in kf.split(X_train_resampled, y_train_resampled):
        X_train_fold, X_val_fold = X_train_resampled.iloc[train_index], X_train_resampled.iloc[val_index]
        y_train_fold, y_val_fold = y_train_resampled[train_index], y_train_resampled[val_index]
    
        y_train_fold_onehot = tf.keras.utils.to_categorical(y_train_fold, num_classes=3)
        y_val_fold_onehot = tf.keras.utils.to_categorical(y_val_fold, num_classes=3)
    
        final_model.fit(
            X_train_fold, y_train_fold_onehot,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            validation_data=(X_val_fold, y_val_fold_onehot),
            callbacks=[early_stopping],
            verbose=1
        )
    
    # inner
    y_pred_proba_val = final_model.predict(X_val_test_scaled)
    composite_metric_val, pairwise_auc_val, global_auc_val = compute_composite_metric(y_val_test.values, y_pred_proba_val)
    print(f"Internal Test Composite Metric: {composite_metric_val:.4f}")
    print(f"Internal Test Pairwise AUC (Class 0 vs Class 1): {pairwise_auc_val:.4f}")
    print(f"Internal Test Global Macro-AUC: {global_auc_val:.4f}")
    
    # outer
    y_pred_proba_external = final_model.predict(X_external_test_scaled)
    composite_metric_external, pairwise_auc_external, global_auc_external = compute_composite_metric(y_external_test.values, y_pred_proba_external)
    print(f"External Test Composite Metric: {composite_metric_external:.4f}")
    print(f"External Test Pairwise AUC (Class 0 vs Class 1): {pairwise_auc_external:.4f}")
    print(f"External Test Global Macro-AUC: {global_auc_external:.4f}")
        
if __name__ == "__main__":
    main()
