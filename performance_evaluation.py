import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from matplotlib.colors import LinearSegmentedColormap
import math
from scipy.stats import norm
import compare_auc_delong_xu
import os

# Use non-interactive backend to avoid GUI and speed up file saving
import matplotlib
matplotlib.use('Agg')

# ensure we have sensible defaults / helpers when running as standalone
labels = ['TRD', 'nTRD', 'HC']

def prob_to_onehot(probs):
    """Convert probability array (n_samples, n_classes) to one-hot labels."""
    if probs is None:
        return None
    idx = np.argmax(probs, axis=1)
    onehot = np.zeros((probs.shape[0], probs.shape[1]), dtype=int)
    onehot[np.arange(probs.shape[0]), idx] = 1
    return onehot

# result directory fallback
result_dir = os.path.join(os.getcwd(), 'visualization_results')
os.makedirs(result_dir, exist_ok=True)

# Try to load commonly-saved numpy arrays if variables are not already defined
def _try_load(name):
    p = os.path.join(os.getcwd(), name)
    return np.load(p) if os.path.exists(p) else None

if 'y_inner_test' not in globals():
    y_inner_test = _try_load('y_inner_test.npy')
if 'y_inner_pred' not in globals():
    y_inner_pred = _try_load('y_inner_pred.npy')
if 'y_inner_pred_onehot' not in globals():
    if y_inner_pred is not None:
        y_inner_pred_onehot = prob_to_onehot(y_inner_pred)
    else:
        y_inner_pred_onehot = _try_load('y_inner_pred_onehot.npy')

if 'y_outer_test' not in globals():
    y_outer_test = _try_load('y_outer_test.npy')
if 'y_outer_pred' not in globals():
    y_outer_pred = _try_load('y_outer_pred.npy')
if 'y_outer_pred_onehot' not in globals():
    if y_outer_pred is not None:
        y_outer_pred_onehot = prob_to_onehot(y_outer_pred)
    else:
        y_outer_pred_onehot = _try_load('y_outer_pred_onehot.npy')



# 根据真实标签和预测概率计算AUC
def calculate_auc(y_true, y_pred_prob, class_idx=None, macro=False):
    """
    计算特定类别或宏平均的AUC
    
    参数:
    y_true: one-hot编码的真实标签
    y_pred_prob: 预测概率
    class_idx: 要计算AUC的类别索引，如果为None且macro为False，则计算所有类别的AUC
    macro: 是否计算宏平均AUC
    
    返回:
    fpr, tpr, roc_auc: ROC曲线的假阳性率、真阳性率和AUC
    """
    n_classes = y_true.shape[1]
    
    # 将真实标签转换为类别索引
    y_true_cat = np.argmax(y_true, axis=1)
    
    if macro:
        # 计算宏平均AUC
        # 对每个类别进行二分类处理
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            # 当前类为正例，其他类为负例
            y_binary = (y_true_cat == i).astype(int)
            # 增加ROC曲线的采样点
            fpr[i], tpr[i], _ = roc_curve(y_binary, y_pred_prob[:, i], drop_intermediate=False)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 计算宏平均 - 使用更多的插值点
        all_fpr = np.linspace(0, 1, 30)  
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
        mean_tpr /= n_classes
        
        macro_auc = auc(all_fpr, mean_tpr)
        return all_fpr, mean_tpr, macro_auc
    
    elif class_idx is not None:
        # 计算特定类别的AUC
        y_binary = (y_true_cat == class_idx).astype(int)
        # 增加ROC曲线的采样点
        fpr, tpr, _ = roc_curve(y_binary, y_pred_prob[:, class_idx], drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    else:
        # 计算每个类别的AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            y_binary = (y_true_cat == i).astype(int)
            # 增加ROC曲线的采样点
            fpr[i], tpr[i], _ = roc_curve(y_binary, y_pred_prob[:, i], drop_intermediate=False)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        return fpr, tpr, roc_auc


# ---- DeLong implementation for AUC variance and CI ----
def compute_midrank(x):
    # compute midranks for tied scores
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(n, dtype=float)
    T2[J] = T + 1.0
    return T2

def fast_delong(preds_sorted_transposed, label_1_count):
    # preds_sorted_transposed shape: (n_classifiers, n_samples)
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    tx = np.zeros((preds_sorted_transposed.shape[0], m))
    ty = np.zeros((preds_sorted_transposed.shape[0], n))
    for r in range(preds_sorted_transposed.shape[0]):
        tx[r, :] = compute_midrank(preds_sorted_transposed[r, :m])
        ty[r, :] = compute_midrank(preds_sorted_transposed[r, m:])
    tx_mean = np.mean(tx, axis=1)
    ty_mean = np.mean(ty, axis=1)
    vx = np.cov(tx, bias=True)
    vy = np.cov(ty, bias=True)
    sx = (vx - np.outer(tx_mean, tx_mean)) / m
    sy = (vy - np.outer(ty_mean, ty_mean)) / n
    var = sx.sum() / (m * m) + sy.sum() / (n * n)
    return var

def delong_auc_variance(y_true_binary, y_scores):
    # y_true_binary: 0/1 array, y_scores: probability for positive class
    # Returns (auc, variance)
    # sort by score descending
    order = np.argsort(-y_scores)
    y_scores_sorted = y_scores[order]
    y_true_sorted = y_true_binary[order]
    pos_count = int(np.sum(y_true_sorted == 1))
    if pos_count == 0 or pos_count == len(y_true_sorted):
        # degenerate case
        return float('nan'), float('nan')
    # concatenate scores: positives first then negatives
    preds_sorted = np.concatenate((y_scores_sorted[y_true_sorted == 1], y_scores_sorted[y_true_sorted == 0]))
    preds_sorted = preds_sorted.reshape(1, -1)
    try:
        var = fast_delong(preds_sorted, pos_count)
    except Exception:
        var = float('nan')
    auc = roc_auc_score(y_true_binary, y_scores)
    return auc, var

def delong_ci(y_true_binary, y_scores, alpha=0.95):
    """
    Compute AUC and DeLong-based (approximate) confidence interval using
    the implementation from yandexdataschool/roc_comparison (compare_auc_delong_xu).

    Returns: (auc, lower_ci, upper_ci). If computation fails or is degenerate,
    returns (auc, nan, nan).
    """
    try:
        # compare_auc_delong_xu.delong_roc_variance expects ground truth as 0/1
        auc_val, delongcov = compare_auc_delong_xu.delong_roc_variance(y_true_binary, y_scores)
        # delongcov is a covariance matrix; variance for the single AUC is delongcov[0,0]
        var = float(delongcov[0, 0]) if delongcov is not None else float('nan')
    except Exception:
        # fallback to previous implementation if anything goes wrong
        try:
            auc_val, var = delong_auc_variance(y_true_binary, y_scores)
        except Exception:
            return float('nan'), float('nan'), float('nan')

    if math.isnan(auc_val) or math.isnan(var) or var <= 0:
        # fallback to bootstrap CI for robustness
        try:
            return bootstrap_auc_ci(y_true_binary, y_scores, n_boot=1000, alpha=alpha)
        except Exception:
            return auc_val, float('nan'), float('nan')

    se = math.sqrt(var)
    z = norm.ppf(1 - (1 - alpha) / 2)
    lower = auc_val - z * se
    upper = auc_val + z * se
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return float(auc_val), float(lower), float(upper)


def stratified_bootstrap_macro_auc(y_true_onehot, y_pred_prob, n_boot=1000, alpha=0.95, random_state=42):
    """Estimate macro AUC and percentile bootstrap CI with stratified resampling.

    Returns: (macro_auc, lower_ci, upper_ci)
    """
    rng = np.random.RandomState(random_state)
    y_true_cat = np.argmax(y_true_onehot, axis=1)
    classes, counts = np.unique(y_true_cat, return_counts=True)
    # compute original macro AUC
    _, _, macro_auc = calculate_auc(y_true_onehot, y_pred_prob, macro=True)

    boot_vals = []
    for i in range(n_boot):
        # build stratified sample indices
        indices = []
        for c in classes:
            cls_idx = np.where(y_true_cat == c)[0]
            if len(cls_idx) == 0:
                continue
            sampled = rng.choice(cls_idx, size=len(cls_idx), replace=True)
            indices.append(sampled)
        indices = np.concatenate(indices)
        # shuffle
        rng.shuffle(indices)
        y_true_bs = y_true_onehot[indices]
        y_pred_bs = y_pred_prob[indices]
        try:
            _, _, bs_auc = calculate_auc(y_true_bs, y_pred_bs, macro=True)
            boot_vals.append(bs_auc)
        except Exception:
            continue

    if len(boot_vals) == 0:
        return macro_auc, float('nan'), float('nan')

    lower = np.percentile(boot_vals, 100 * (1 - alpha) / 2)
    upper = np.percentile(boot_vals, 100 * (1 + alpha) / 2)
    return float(macro_auc), float(lower), float(upper)


def bootstrap_auc_ci(y_true_binary, y_scores, n_boot=1000, alpha=0.95, random_state=42):
    """Compute percentile bootstrap CI for binary AUC.

    Returns: (auc, lower_ci, upper_ci)
    """
    rng = np.random.RandomState(random_state)
    try:
        auc_orig = float(roc_auc_score(y_true_binary, y_scores))
    except Exception:
        return float('nan'), float('nan'), float('nan')

    n = len(y_true_binary)
    boot_vals = []
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            val = roc_auc_score(y_true_binary[idx], y_scores[idx])
            boot_vals.append(val)
        except Exception:
            continue

    if len(boot_vals) == 0:
        return auc_orig, float('nan'), float('nan')

    lower = np.percentile(boot_vals, 100 * (1 - alpha) / 2)
    upper = np.percentile(boot_vals, 100 * (1 + alpha) / 2)
    return auc_orig, float(lower), float(upper)

# 创建三分类混淆矩阵
def create_confusion_matrix(y_true, y_pred, title, filename, normalize=False):
    y_true_cat = np.argmax(y_true, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_cat, y_pred_cat)
    
    # 归一化混淆矩阵（按行）
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
        title += " (归一化)"
    else:
        cm_display = cm
        fmt = 'd'
    
    df_cm = pd.DataFrame(cm_display, index=labels, columns=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt=fmt, cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=300)
    plt.close()
    
    return cm

# 计算二分类评估指标（1类 vs 其他）
def calculate_binary_metrics(y_true, y_pred, focus_class_idx):
    """计算二分类评估指标 (一类对其他)
    
    参数:
    y_true: one-hot编码的真实标签
    y_pred: one-hot编码的预测标签
    focus_class_idx: 关注的类别索引
    
    返回:
    metrics: 包含准确率、精确率、召回率（敏感度）、特异度和F1分数的字典
    """
    y_true_cat = np.argmax(y_true, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    
    # 将多分类转换为二分类 (目标类 vs 其他类)
    y_true_binary = (y_true_cat == focus_class_idx).astype(int)
    y_pred_binary = (y_pred_cat == focus_class_idx).astype(int)
    
    # 计算混淆矩阵元素
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

# 计算多分类和二分类的评估指标
def calculate_metrics(y_true, y_pred, dataset_name):
    """计算多分类和二分类的评估指标
    
    参数:
    y_true: one-hot编码的真实标签
    y_pred: one-hot编码的预测标签
    dataset_name: 数据集名称
    
    返回:
    metrics_dict: 包含多分类和二分类评估指标的字典
    """
    y_true_cat = np.argmax(y_true, axis=1)
    y_pred_cat = np.argmax(y_pred, axis=1)
    
    # 多分类指标
    accuracy = accuracy_score(y_true_cat, y_pred_cat)
    precision_macro = precision_score(y_true_cat, y_pred_cat, average='macro')
    recall_macro = recall_score(y_true_cat, y_pred_cat, average='macro')
    f1_macro = f1_score(y_true_cat, y_pred_cat, average='macro')
    
    precision_weighted = precision_score(y_true_cat, y_pred_cat, average='weighted')
    recall_weighted = recall_score(y_true_cat, y_pred_cat, average='weighted')
    f1_weighted = f1_score(y_true_cat, y_pred_cat, average='weighted')
    
    # 每个类别的指标
    class_report = classification_report(y_true_cat, y_pred_cat, target_names=labels, output_dict=True)
    
    # 二分类指标（每个类别 vs 其他）
    binary_metrics = {}
    for i, label in enumerate(labels):
        binary_metrics[label] = calculate_binary_metrics(y_true, y_pred, i)
    
    # 打印结果
    print(f"\n{dataset_name} Evaluation Metrics:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Average Precision: {precision_macro:.4f}")
    print(f"Macro Average Recall: {recall_macro:.4f}")
    print(f"Macro Average F1 Score: {f1_macro:.4f}")
    
    print("\nWeighted Average:")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    
    print("\nMulti-class Metrics by Category:")
    for label in labels:
        print(f"\n{label} Class:")
        print(f"Precision: {class_report[label]['precision']:.4f}")
        print(f"Recall: {class_report[label]['recall']:.4f}")
        print(f"F1 Score: {class_report[label]['f1-score']:.4f}")
        print(f"Sample Count: {class_report[label]['support']}")
    
    print("\nBinary Classification Metrics (One-vs-Others):")
    for label in labels:
        metrics = binary_metrics[label]
        print(f"\n{label} vs Others:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall/Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
    
    return {
        'dataset': dataset_name,
        'overall_accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'class_metrics': class_report,
        'binary_metrics': binary_metrics
    }

# Create combined visualization including AUC curves and confusion matrices
def create_combined_visualization(y_true_inner, y_pred_inner, y_true_outer, y_pred_outer):
    # Set global matplotlib style for publication quality
    plt.style.use('seaborn-v0_8-white')
    
    # 设置精美的字体和专业排版
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.weight': 'regular',
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
        'figure.dpi': 400,
        'savefig.dpi': 800,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'axes.axisbelow': True,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })
    
    # 科研期刊品质配色方案 - 现代美学风格配色
    color_palette = {
        'trd': '#FF6B6B',          # 柔和珊瑚红 (新配色)
        'macro': '#4ECDC4',        # 青绿松石色 (新配色)
        'trd_ntrd': '#9B59B6',     # 紫色 - TRD vs nTRD
        'trd_hc': '#F39C12',       # 橙色 - TRD vs HC
        'reference': '#555B6E',    # 深蓝灰色 (新配色)
        'background': '#FFFFFF',   # 淡蓝白色背景 (新配色)
        'grid': '#DCDFE6',         # 浅蓝灰网格线 (新配色)
        'border': '#2C3E50',       # 深青蓝色边框 (新配色)
        'shadow': '#BDC3C7',       # 浅灰阴影色 (新配色)
        'text': '#2C3E50',         # 深青蓝色文本 (新配色)
        'title': '#16697A',        # 深青绿色标题 (新配色)
    }
    
    # 创建自定义调色板用于混淆矩阵 - 高端优雅配色
    custom_cm = sns.light_palette("#5E72E4", n_colors=100, as_cmap=True)  # 渐变蓝紫色调
    
    # 创建带有淡色背景的画布
    fig = plt.figure(figsize=(18, 15), facecolor=color_palette['background'])
    
    # 定义子图相对位置和大小(底部，左侧，宽度，高度)
    # 复杂的网格布局，以获得更精确的控制和边距
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                         left=0.07, right=0.96, bottom=0.07, top=0.92, 
                         wspace=0.25, hspace=0.25)
    
    # 创建子图
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 添加子图标注 a) b) c) d)
    ax1.text(-0.12, 1.08, 'a)', transform=ax1.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')
    ax2.text(-0.12, 1.08, 'b)', transform=ax2.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')
    ax3.text(-0.20, 1.08, 'c)', transform=ax3.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')
    ax4.text(-0.20, 1.08, 'd)', transform=ax4.transAxes, fontsize=30, fontweight='bold', va='top', ha='left')
    
    # 添加3D效果的标题和边框阴影
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(color_palette['background'])
        # 添加轻微阴影效果
        for spine in ax.spines.values():
            spine.set_edgecolor(color_palette['border'])
            spine.set_linewidth(1.2)
    
    # 主标题
    #fig.suptitle('Deep Learning Model Performance Evaluation', 
    #             fontsize=22, y=0.98, fontweight='bold', color=color_palette['title'])
    
    # ===== 1. Internal Test Set ROC Curve =====
    # Calculate TRD vs Others AUC
    fpr_inner_trd, tpr_inner_trd, roc_auc_inner_trd = calculate_auc(y_true_inner, y_pred_inner, class_idx=0)
    ax1.plot(fpr_inner_trd, tpr_inner_trd, lw=3.0, color=color_palette['trd'], 
             path_effects=[PathEffects.withStroke(linewidth=4, foreground=color_palette['trd'] + '30')],
             label=f'TRD vs Others (AUC = {roc_auc_inner_trd:.3f})')
    
    # Calculate Macro average AUC
    fpr_inner_macro, tpr_inner_macro, roc_auc_inner_macro = calculate_auc(y_true_inner, y_pred_inner, macro=True)
    ax1.plot(fpr_inner_macro, tpr_inner_macro, lw=3.0, color=color_palette['macro'], 
             linestyle='--', path_effects=[PathEffects.withStroke(linewidth=4, foreground=color_palette['macro'] + '30')],
             label=f'Macro Average (AUC = {roc_auc_inner_macro:.3f})')
    
    # Calculate TRD vs nTRD AUC (1-on-1)
    inner_trd_ntrd_metrics = calculate_one_on_one_metrics(y_true_inner, y_pred_inner, 0, 1, 'TRD', 'nTRD', 'Internal')
    if inner_trd_ntrd_metrics:
        ax1.plot(inner_trd_ntrd_metrics['fpr'], inner_trd_ntrd_metrics['tpr'], lw=2.5, 
                color=color_palette['trd_ntrd'], linestyle='-.', 
                label=f'TRD vs nTRD (AUC = {inner_trd_ntrd_metrics["auc"]:.3f})')
    
    # Calculate TRD vs HC AUC (1-on-1)
    inner_trd_hc_metrics = calculate_one_on_one_metrics(y_true_inner, y_pred_inner, 0, 2, 'TRD', 'HC', 'Internal')
    if inner_trd_hc_metrics:
        ax1.plot(inner_trd_hc_metrics['fpr'], inner_trd_hc_metrics['tpr'], lw=2.5, 
                color=color_palette['trd_hc'], linestyle=':', 
                label=f'TRD vs HC (AUC = {inner_trd_hc_metrics["auc"]:.3f})')
    
    # Reference line with improved style
    ax1.plot([0, 1], [0, 1], color=color_palette['reference'], linestyle='--', lw=1.5, alpha=0.6)
    
    # Fill area under curves with gradient for 3D effect
    # 为TRD曲线创建优雅的渐变填充
    for i in range(len(fpr_inner_trd)-1):
        ax1.fill_between(fpr_inner_trd[i:i+2], tpr_inner_trd[i:i+2], alpha=0.15-0.1*(i/len(fpr_inner_trd)), 
                          color=color_palette['trd'])
    
    # 高级造型
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontweight='bold', labelpad=10)
    ax1.set_ylabel('True Positive Rate', fontweight='bold', labelpad=10)
    #ax1.set_title('Internal Test Set - ROC Analysis', fontweight='bold', pad=15, color=color_palette['title'])
    ax1.grid(True, linestyle='--', alpha=0.3, color=color_palette['grid'])
    
    # 移除顶部和右侧边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 添加精美的图例
    leg1 = ax1.legend(loc="lower right", frameon=True, framealpha=0.85, 
              edgecolor='lightgray', facecolor='white', title="ROC Curves")
    leg1.get_title().set_fontweight('bold')
    
    
    # ===== 2. External Test Set ROC Curve =====
    # Calculate TRD vs Others AUC
    fpr_outer_trd, tpr_outer_trd, roc_auc_outer_trd = calculate_auc(y_true_outer, y_pred_outer, class_idx=0)
    ax2.plot(fpr_outer_trd, tpr_outer_trd, lw=3.0, color=color_palette['trd'], 
             path_effects=[PathEffects.withStroke(linewidth=4, foreground=color_palette['trd'] + '30')],
             label=f'TRD vs Others (AUC = {roc_auc_outer_trd:.3f})')
    
    # Calculate Macro average AUC
    fpr_outer_macro, tpr_outer_macro, roc_auc_outer_macro = calculate_auc(y_true_outer, y_pred_outer, macro=True)
    ax2.plot(fpr_outer_macro, tpr_outer_macro, lw=3.0, color=color_palette['macro'], 
             linestyle='--', path_effects=[PathEffects.withStroke(linewidth=4, foreground=color_palette['macro'] + '30')],
             label=f'Macro Average (AUC = {roc_auc_outer_macro:.3f})')
    
    # Calculate TRD vs nTRD AUC (1-on-1)
    outer_trd_ntrd_metrics = calculate_one_on_one_metrics(y_true_outer, y_pred_outer, 0, 1, 'TRD', 'nTRD', 'External')
    if outer_trd_ntrd_metrics:
        ax2.plot(outer_trd_ntrd_metrics['fpr'], outer_trd_ntrd_metrics['tpr'], lw=2.5, 
                color=color_palette['trd_ntrd'], linestyle='-.', 
                label=f'TRD vs nTRD (AUC = {outer_trd_ntrd_metrics["auc"]:.3f})')
    
    # Calculate TRD vs HC AUC (1-on-1)
    outer_trd_hc_metrics = calculate_one_on_one_metrics(y_true_outer, y_pred_outer, 0, 2, 'TRD', 'HC', 'External')
    if outer_trd_hc_metrics:
        ax2.plot(outer_trd_hc_metrics['fpr'], outer_trd_hc_metrics['tpr'], lw=2.5, 
                color=color_palette['trd_hc'], linestyle=':', 
                label=f'TRD vs HC (AUC = {outer_trd_hc_metrics["auc"]:.3f})')
    
    # Reference line with improved style
    ax2.plot([0, 1], [0, 1], color=color_palette['reference'], linestyle='--', lw=1.5, alpha=0.6)
    
    # Fill area under curves with gradient for 3D effect
    # 应用渐变填充
    for i in range(len(fpr_outer_trd)-1):
        ax2.fill_between(fpr_outer_trd[i:i+2], tpr_outer_trd[i:i+2], alpha=0.15-0.1*(i/len(fpr_outer_trd)), 
                          color=color_palette['trd'])
    
    # 高级造型
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontweight='bold', labelpad=10)
    ax2.set_ylabel('True Positive Rate', fontweight='bold', labelpad=10)
    #ax2.set_title('External Test Set - ROC Analysis', fontweight='bold', pad=15, color=color_palette['title'])
    ax2.grid(True, linestyle='--', alpha=0.3, color=color_palette['grid'])
    
    # 移除顶部和右侧边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 添加精美的图例
    leg2 = ax2.legend(loc="lower right", frameon=True, framealpha=0.85, 
              edgecolor='lightgray', facecolor='white', title="ROC Curves")
    leg2.get_title().set_fontweight('bold')
    
    # ===== 3. Internal Test Set Confusion Matrix =====
    # Calculate confusion matrix
    cm_inner = confusion_matrix(np.argmax(y_true_inner, axis=1), np.argmax(prob_to_onehot(y_pred_inner), axis=1))
    cm_df = pd.DataFrame(cm_inner, index=labels, columns=labels)
    
    # 创建带有白色边框效果的热图，增强3D感 - 调整文本颜色为黑色以提高可读性
    sns.heatmap(cm_df, annot=True, fmt='d', cmap=custom_cm, ax=ax3, 
                annot_kws={"size": 16, "weight": "bold", "color": "black"}, 
                cbar_kws={"shrink": 0.75, "label": "Count", "pad": 0.01},
                linewidths=1.0, linecolor='white', square=True)
    
    # 增强3D效果
    for _, spine in ax3.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('#FFFFFF')
    
    # 添加阴影效果和边框
    #ax3.set_title('Internal Test Set - Confusion Matrix', fontweight='bold', pad=15, color=color_palette['title'])
    ax3.set_ylabel('True Label', fontweight='bold', labelpad=10)
    ax3.set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
    
    # 突出显示准确率
    #accuracy_inner = np.trace(cm_inner) / np.sum(cm_inner)
    #ax3.text(0.5, -0.2, f"Accuracy: {accuracy_inner:.3f}", 
    #         transform=ax3.transAxes, fontsize=13, horizontalalignment='center',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
    #                  edgecolor=color_palette['border'], linewidth=1))
    
    # ===== 4. External Test Set Confusion Matrix =====
    # Calculate confusion matrix
    cm_outer = confusion_matrix(np.argmax(y_true_outer, axis=1), np.argmax(prob_to_onehot(y_pred_outer), axis=1))
    cm_df = pd.DataFrame(cm_outer, index=labels, columns=labels)
    
    # 创建带有白色边框效果的热图，增强3D感 - 调整文本颜色为黑色以提高可读性
    sns.heatmap(cm_df, annot=True, fmt='d', cmap=custom_cm, ax=ax4, 
                annot_kws={"size": 16, "weight": "bold", "color": "black"}, 
                cbar_kws={"shrink": 0.75, "label": "Count", "pad": 0.01},
                linewidths=1.0, linecolor='white', square=True)
    
    # 增强3D效果
    for _, spine in ax4.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('#FFFFFF')
    
    # 添加阴影效果和边框
    #ax4.set_title('External Test Set - Confusion Matrix', fontweight='bold', pad=15, color=color_palette['title'])
    ax4.set_ylabel('True Label', fontweight='bold', labelpad=10)
    ax4.set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
    
    # 突出显示准确率
    #accuracy_outer = np.trace(cm_outer) / np.sum(cm_outer)
    #ax4.text(0.5, -0.2, f"Accuracy: {accuracy_outer:.3f}", 
    #         transform=ax4.transAxes, fontsize=13, horizontalalignment='center',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
    #                  edgecolor=color_palette['border'], linewidth=1))
    
    # 调整布局以避免重叠
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    # 保存高分辨率格式
    plt.savefig(os.path.join(result_dir, 'combined_visualization_advanced.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(result_dir, 'combined_visualization_advanced.svg'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(result_dir, 'combined_visualization_advanced.png'), dpi=150, bbox_inches='tight')
    
    # 保存带水印的高分辨率版本
    plt.savefig(os.path.join(result_dir, 'combined_visualization_advanced_copy.png'), dpi=150, bbox_inches='tight')
    
    # 创建透明背景版本
    fig.patch.set_alpha(0.0)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.patch.set_alpha(0.0)
    plt.savefig(os.path.join(result_dir, 'combined_visualization_advanced_transparent.png'), 
                dpi=150, bbox_inches='tight', transparent=True)
    
    # 在终端中提供描述性反馈
    print("Created journal-level advanced visualization:")
    print(f"- Vector format: '{os.path.join(result_dir, 'combined_visualization_advanced.svg')}', '{os.path.join(result_dir, 'combined_visualization_advanced.pdf')}'")
    print(f"- High-resolution bitmap: '{os.path.join(result_dir, 'combined_visualization_advanced.png')}'")
    print(f"- Copy version: '{os.path.join(result_dir, 'combined_visualization_advanced_copy.png')}'")
    print(f"- Transparent background version: '{os.path.join(result_dir, 'combined_visualization_advanced_transparent.png')}'")
    print("Features: 3D effect, journal-level layout, professional color scheme, high-resolution output (800 DPI)")
    
    # 关闭图形以释放内存
    plt.close()

# 计算约登指数 (Youden's Index)
def calculate_youden_index(sensitivity, specificity):
    """计算约登指数 (敏感度 + 特异度 - 1)"""
    return sensitivity + specificity - 1

def calculate_one_on_one_metrics(y_true, y_pred_prob, class1_idx, class2_idx, class1_name, class2_name, dataset_name):
    """计算1对1的二分类指标"""
    # 获取只包含两个特定类别的样本索引
    true_labels = np.argmax(y_true, axis=1)
    mask = (true_labels == class1_idx) | (true_labels == class2_idx)
    
    if np.sum(mask) == 0:
        return None
    
    # 提取相关样本
    y_true_binary = true_labels[mask]
    y_pred_prob_binary = y_pred_prob[mask]
    
    # 重新映射标签：class1 -> 1, class2 -> 0
    y_true_binary = (y_true_binary == class1_idx).astype(int)
    
    # 使用class1的预测概率作为正类概率
    y_pred_prob_class1 = y_pred_prob_binary[:, class1_idx]
    
    # 计算AUC
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    auc_score = roc_auc_score(y_true_binary, y_pred_prob_class1)
    
    # 使用最优阈值进行预测（基于约登指数）
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_prob_class1)
    youden_scores = tpr - fpr
    optimal_idx = np.argmax(youden_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # 基于最优阈值的预测
    y_pred_binary = (y_pred_prob_class1 >= optimal_threshold).astype(int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # 计算各种指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    youden_index = sensitivity + specificity - 1
    
    return {
        'dataset': dataset_name,
        'comparison': f"{class1_name} vs {class2_name}",
        'sample_count': np.sum(mask),
        'class1_count': np.sum(y_true_binary == 1),
        'class2_count': np.sum(y_true_binary == 0),
        'auc': auc_score,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'youden_index': youden_index,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds
    }

def create_one_on_one_roc_curves(y_inner_test, y_inner_pred, y_outer_test, y_outer_pred):
    """创建1对1的ROC曲线，特别关注TRD vs nTRD和TRD vs HC"""
    
    # 定义要绘制的比较对
    comparisons = [
        (0, 1, 'TRD', 'nTRD'),  # TRD vs nTRD
        (0, 2, 'TRD', 'HC'),    # TRD vs HC
        (1, 2, 'nTRD', 'HC')    # nTRD vs HC
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (class1_idx, class2_idx, class1_name, class2_name) in enumerate(comparisons):
        # 内部测试集
        ax_inner = axes[0, i]
        inner_metrics = calculate_one_on_one_metrics(
            y_inner_test, y_inner_pred, class1_idx, class2_idx, 
            class1_name, class2_name, "Internal Test"
        )
        
        if inner_metrics:
            ax_inner.plot(inner_metrics['fpr'], inner_metrics['tpr'], 
                         linewidth=3, label=f'AUC = {inner_metrics["auc"]:.3f}',
                         color='blue')
            ax_inner.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax_inner.set_xlim([0.0, 1.0])
            ax_inner.set_ylim([0.0, 1.05])
            ax_inner.set_xlabel('False Positive Rate')
            ax_inner.set_ylabel('True Positive Rate')
            ax_inner.set_title(f'Internal: {class1_name} vs {class2_name}\n'
                              f'(n={inner_metrics["sample_count"]})')
            ax_inner.legend(loc="lower right")
            ax_inner.grid(True, alpha=0.3)
        
        # 外部测试集
        ax_outer = axes[1, i]
        outer_metrics = calculate_one_on_one_metrics(
            y_outer_test, y_outer_pred, class1_idx, class2_idx, 
            class1_name, class2_name, "External Test"
        )
        
        if outer_metrics:
            ax_outer.plot(outer_metrics['fpr'], outer_metrics['tpr'], 
                         linewidth=3, label=f'AUC = {outer_metrics["auc"]:.3f}',
                         color='red')
            ax_outer.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax_outer.set_xlim([0.0, 1.0])
            ax_outer.set_ylim([0.0, 1.05])
            ax_outer.set_xlabel('False Positive Rate')
            ax_outer.set_ylabel('True Positive Rate')
            ax_outer.set_title(f'External: {class1_name} vs {class2_name}\n'
                              f'(n={outer_metrics["sample_count"]})')
            ax_outer.legend(loc="lower right")
            ax_outer.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'one_on_one_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(result_dir, 'one_on_one_roc_curves.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparisons

# 主流程
# 1. 创建综合可视化
print("Creating combined visualization...")
create_combined_visualization(y_inner_test, y_inner_pred, y_outer_test, y_outer_pred)

# 2. 创建1对1 ROC曲线
print("Creating 1-on-1 ROC curves...")
comparisons = create_one_on_one_roc_curves(y_inner_test, y_inner_pred, y_outer_test, y_outer_pred)

# 不再需要单独生成混淆矩阵，跳过这部分代码
print("Calculating evaluation metrics...")
inner_metrics = calculate_metrics(y_inner_test, y_inner_pred_onehot, "Internal Test")
outer_metrics = calculate_metrics(y_outer_test, y_outer_pred_onehot, "External Test")

# 4. 准备简化版的结果数据框
results = []

# 添加整体指标和二分类指标（一对多）
for metrics in [inner_metrics, outer_metrics]:
    # 计算总体特异度为各类别特异度的平均值
    overall_specificity = np.mean([metrics['binary_metrics'][label]['specificity'] for label in labels])
    # 计算总体约登指数
    overall_youden = calculate_youden_index(metrics['macro_recall'], overall_specificity)
    
    # 总体多分类指标
    overall_row = {
        'Dataset': 'Internal Test' if metrics['dataset'] == 'Internal Test' else 'External Test',
        'Category': 'Overall',
        'Accuracy': metrics['overall_accuracy'],
        'Sensitivity/Recall': metrics['macro_recall'],
        'Specificity': overall_specificity,
        'Precision': metrics['macro_precision'],
        'Youden Index': overall_youden,
        'F1 Score': metrics['macro_f1']
    }
    # compute macro AUC + bootstrap CI (stratified)
    try:
        if metrics['dataset'] == 'Internal Test':
            y_true_all = y_inner_test
            y_pred_prob_all = y_inner_pred
        else:
            y_true_all = y_outer_test
            y_pred_prob_all = y_outer_pred

        if y_true_all is not None and y_pred_prob_all is not None:
            macro_auc, lower_m, upper_m = stratified_bootstrap_macro_auc(
                y_true_all, y_pred_prob_all, n_boot=1000, alpha=0.95)
            overall_row['AUC'] = f"{macro_auc:.3f} ({lower_m:.3f}-{upper_m:.3f})"
        else:
            overall_row['AUC'] = ''
    except Exception:
        overall_row['AUC'] = ''

    results.append(overall_row)
    
    # 二分类指标（一对多）
    for label in labels:
        binary_metrics = metrics['binary_metrics'][label]
        youden_index = calculate_youden_index(binary_metrics['sensitivity'], binary_metrics['specificity'])
        # compute AUC + 95% CI (DeLong) for this one-vs-others task if possible
        auc_str = ''
        try:
            # y_true arrays used elsewhere are one-hot; reconstruct binary true labels
            if metrics['dataset'] == 'Internal Test':
                y_true_all = y_inner_test
                y_pred_prob_all = y_inner_pred
            else:
                y_true_all = y_outer_test
                y_pred_prob_all = y_outer_pred

            if y_true_all is not None and y_pred_prob_all is not None:
                y_true_binary = (np.argmax(y_true_all, axis=1) == labels.index(label)).astype(int)
                y_scores = y_pred_prob_all[:, labels.index(label)]
                try:
                    auc_val, lower_ci, upper_ci = delong_ci(y_true_binary, y_scores, alpha=0.95)
                    if not math.isnan(auc_val):
                        auc_str = f"{auc_val:.3f} ({lower_ci:.3f}-{upper_ci:.3f})"
                except Exception:
                    auc_str = ''
        except Exception:
            auc_str = ''

        binary_row = {
            'Dataset': 'Internal Test' if metrics['dataset'] == 'Internal Test' else 'External Test',
            'Category': f"{label} vs Others",
            'Accuracy': binary_metrics['accuracy'],
            'Sensitivity/Recall': binary_metrics['sensitivity'],
            'Specificity': binary_metrics['specificity'],
            'Precision': binary_metrics['precision'],
            'Youden Index': youden_index,
            'F1 Score': binary_metrics['f1']
        }
        # include AUC/CI if computed
        binary_row['AUC'] = auc_str
        results.append(binary_row)

# 添加1对1的二分类指标
print("Calculating 1-on-1 metrics...")
for y_test, y_pred, dataset_name in [(y_inner_test, y_inner_pred, "Internal Test"), 
                                     (y_outer_test, y_outer_pred, "External Test")]:
    for class1_idx, class2_idx, class1_name, class2_name in comparisons:
        one_on_one_metrics = calculate_one_on_one_metrics(
            y_test, y_pred, class1_idx, class2_idx, 
            class1_name, class2_name, dataset_name
        )
        
        if one_on_one_metrics:
            one_on_one_row = {
                'Dataset': one_on_one_metrics['dataset'],
                'Category': f"{class1_name} vs {class2_name} (1-on-1)",
                'Accuracy': one_on_one_metrics['accuracy'],
                'Sensitivity/Recall': one_on_one_metrics['sensitivity'],
                'Specificity': one_on_one_metrics['specificity'],
                'Precision': one_on_one_metrics['precision'],
                'Youden Index': one_on_one_metrics['youden_index'],
                'F1 Score': one_on_one_metrics['f1'],
                # Compute DeLong CI for 1-on-1 AUC and format
                'AUC': None,
                'Sample Count': one_on_one_metrics['sample_count']
            }
            # compute CI
            try:
                # prepare binary y_true and scores
                y_true_bin = one_on_one_metrics['class1_count'] is not None and one_on_one_metrics['class2_count'] is not None
                # We can reconstruct the binary arrays from the full arrays used earlier
                if one_on_one_metrics['dataset'] == 'Internal Test':
                    y_true_full = y_inner_test
                    y_pred_full = y_inner_pred
                else:
                    y_true_full = y_outer_test
                    y_pred_full = y_outer_pred

                if y_true_full is not None and y_pred_full is not None:
                    # mask to class1/class2
                    mask = np.isin(np.argmax(y_true_full, axis=1), [class1_idx, class2_idx])
                    y_true_binary = (np.argmax(y_true_full[mask], axis=1) == class1_idx).astype(int)
                    y_scores = y_pred_full[mask][:, class1_idx]
                    try:
                        auc_val, lower_ci, upper_ci = delong_ci(y_true_binary, y_scores, alpha=0.95)
                        if not math.isnan(auc_val):
                            one_on_one_row['AUC'] = f"{auc_val:.3f} ({lower_ci:.3f}-{upper_ci:.3f})"
                        else:
                            one_on_one_row['AUC'] = f"{one_on_one_metrics['auc']:.3f}"
                    except Exception:
                        one_on_one_row['AUC'] = f"{one_on_one_metrics['auc']:.3f}"
                else:
                    one_on_one_row['AUC'] = f"{one_on_one_metrics['auc']:.3f}"
            except Exception:
                one_on_one_row['AUC'] = f"{one_on_one_metrics['auc']:.3f}"

            results.append(one_on_one_row)

# 创建DataFrame并保存为CSV
df_results = pd.DataFrame(results)
csv_file = os.path.join(result_dir, 'simplified_evaluation_results.csv')
df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')

print(f"\nEvaluation completed! Results saved to directory: {result_dir}")
print(f"Simplified evaluation results saved to: {csv_file}")
print(f"Combined visualization results saved to: {os.path.join(result_dir, 'combined_visualization_advanced.png')}")
print(f"1-on-1 ROC curves saved to: {os.path.join(result_dir, 'one_on_one_roc_curves.png')}")

# 将创建文件的代码修改为使用os.path.join
def save_figure(filename_base, dpi=800, save_copy=True, make_transparent=True):
    """保存图形为多种格式，添加文件夹路径"""
    formats = ['.svg', '.pdf', '.png']
    saved_files = []
    
    for fmt in formats:
        filename = os.path.join(result_dir, f"{filename_base}{fmt}")
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        saved_files.append(filename)
    
    if save_copy:
        copy_filename = os.path.join(result_dir, f"{filename_base}_copy.png")
        plt.savefig(copy_filename, dpi=dpi, bbox_inches='tight')
        saved_files.append(copy_filename)
    
    if make_transparent:
        transparent_filename = os.path.join(result_dir, f"{filename_base}_transparent.png")
        plt.savefig(transparent_filename, dpi=dpi, bbox_inches='tight', transparent=True)
        saved_files.append(transparent_filename)
    
    return saved_files 
