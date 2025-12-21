import os
import numpy as np
import pandas as pd
import tensorflow
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


model = tensorflow.keras.models.load_model('final_model.keras')
print("Model input shape:", model.input_shape)

X_train = np.load('X_train.npy', allow_pickle=True)
X_test = np.load('X_outer_test.npy', allow_pickle=True)  
y_test = np.load('y_outer_test.npy', allow_pickle=True)  
lable = np.load('lable.npy', allow_pickle=True)  

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
explainer = shap.KernelExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
np.save("shap.npy", shap_values)
#shap_values = np.load('shap.npy', allow_pickle=True)  
shap_values_nocov=shap_values[:,8:,:]
lable_nocov=lable[8:]
X_nocov=X_test[:, 8:]


def calculate_shap_stats(shap_values, features, feature_names, class_names):
    num_features = shap_values.shape[1]
    num_classes = shap_values.shape[2]

    # Initialize matrices for mean and covariance
    shap_mean_abs = np.zeros((num_features, num_classes))
    shap_covariance = np.zeros((num_features, num_classes))

    for feature_idx in range(num_features):
        for class_idx in range(num_classes):
            shap_vals = shap_values[:, feature_idx, class_idx]
            feat_vals = features[:, feature_idx]

            # Calculate mean of absolute SHAP values
            shap_mean_abs[feature_idx, class_idx] = np.mean(np.abs(shap_vals))

            # Calculate covariance between feature values and SHAP values
            shap_covariance[feature_idx, class_idx] = np.cov(feat_vals, shap_vals)[0, 1]

    df_shap_mean_abs = pd.DataFrame(shap_mean_abs, index=feature_names, columns=class_names)
    df_shap_covariance = pd.DataFrame(shap_covariance, index=feature_names, columns=class_names)

    return df_shap_mean_abs, df_shap_covariance


mean_abs, covariance = calculate_shap_stats(shap_values_nocov, X_nocov,
                                            lable_nocov, ['TRD','nTRD','HC'])

real_shap = mean_abs.copy()
for feature in real_shap.index:
    for cls in real_shap.columns:
        if covariance.loc[feature, cls] < 0:
            real_shap.loc[feature, cls] = -real_shap.loc[feature, cls]

real_shap.to_csv("real_shap.csv")


