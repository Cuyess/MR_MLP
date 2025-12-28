import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patheffects as PathEffects

# ==========================================
# 1. 准备工作 & 风格定义
# ==========================================

# 设置全局字体
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# [混淆矩阵配色] 严格执行您要求的蓝紫色 (#5E72E4)
custom_cm = sns.light_palette("#5E72E4", n_colors=100, as_cmap=True)

# [ROC 曲线配色] Nature (NPG) 经典配色
roc_colors = {
    'TRD vs Others': "#CA2026",    
    'TRD vs nTRD':   "#F79420",    
    'TRD vs HC':     "#838383",  
    'Macro Average': "#38459B",   
}

# 加载数据
try:
    roc_data = np.load('roc_curves_data.npy', allow_pickle=True).item()
    df_cm = pd.read_csv('confusion_matrix_stats.csv')
except FileNotFoundError:
    print("错误：找不到数据文件。请确保 roc_curves_data.npy 和 confusion_matrix_stats.csv 在当前目录下。")
    # 为了演示代码逻辑，创建空结构
    roc_data = {'Internal': {}, 'External': {}}
    df_cm = pd.DataFrame()

mean_fpr = np.linspace(0, 1, 100)

# ==========================================
# 2. 绘图函数定义
# ==========================================

def plot_real_roc_with_ci(ax, dataset_key, title_text):
    """
    绘制 ROC 曲线：带真实阴影，Nature 配色
    """
    data_dict = roc_data.get(dataset_key, {})
    
    # 曲线配置 (名称, 线型, 线宽, Z轴层级)
    curves_config = [
        ('TRD vs HC', ':', 1.5, 2),
        ('TRD vs nTRD', '-.', 1.5, 3),
        ('Macro Average', '--', 2.0, 4),
        ('TRD vs Others', '-', 2.5, 5)
    ]
    
    for name, style, lw, zorder in curves_config:
        if name in data_dict and len(data_dict[name]) > 0:
            tprs = np.array(data_dict[name])
            
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            mean_tpr[-1] = 1.0 
            
            mean_auc = np.mean([np.trapz(t, mean_fpr) for t in tprs])
            color = roc_colors.get(name, '#333333')
            
            # 阴影
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.15, zorder=zorder-1)
            
            # 主曲线 + 白色描边
            ax.plot(mean_fpr, mean_tpr, color=color, linestyle=style, 
                    label=f'{name} (AUC = {mean_auc:.3f})', lw=lw, zorder=zorder,
                    path_effects=[PathEffects.withStroke(linewidth=lw+2, foreground='white', alpha=0.8)])

    # 对角线
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.0, color='#999999', alpha=0.6, zorder=1)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontweight='bold', labelpad=8)
    ax.set_ylabel('True Positive Rate', fontweight='bold', labelpad=8)
    ax.set_title(title_text, fontweight='bold', pad=12, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4, color='#999999')
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelspacing=0.4)

def plot_styled_cm_mean_only(ax, dataset_name, title_text):
    """
    绘制混淆矩阵：[修改点] 只显示均值
    """
    if df_cm.empty: return

    subset = df_cm[df_cm['Dataset'] == dataset_name]
    labels_order = ['TRD', 'nTRD', 'HC']
    
    mean_matrix = np.zeros((3, 3))
    text_matrix = np.empty((3, 3), dtype=object)
    
    for i, true_l in enumerate(labels_order):
        for j, pred_l in enumerate(labels_order):
            row = subset[(subset['True Label'] == true_l) & (subset['Predicted Label'] == pred_l)]
            if not row.empty:
                val_mean = row['Mean'].values[0]
                mean_matrix[i, j] = val_mean
                # [关键修改] 只保留均值，保留1位小数
                text_matrix[i, j] = f"{val_mean:.1f}"
    
    cm_df = pd.DataFrame(mean_matrix, index=labels_order, columns=labels_order)
    
    # 绘制热图 (保留您喜欢的样式)
    sns.heatmap(cm_df, annot=text_matrix, fmt='', cmap=custom_cm, ax=ax, 
                annot_kws={"size": 12, "weight": "bold", "color": "black", "family": "Arial"}, 
                cbar_kws={"shrink": 0.75, "label": "Count (Mean)", "pad": 0.04},
                linewidths=2.0, linecolor='white', square=True)
    
    # 3D 边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_edgecolor('#FFFFFF') 
    
    ax.set_ylabel('True Label', fontweight='bold', labelpad=8, fontsize=10)
    ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=8, fontsize=10)
    ax.set_title(title_text, fontweight='bold', pad=12, fontsize=12)
    ax.set_xticklabels(labels_order, rotation=0)
    ax.set_yticklabels(labels_order, rotation=0)

# ==========================================
# 3. 主画布布局
# ==========================================

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                     left=0.08, right=0.95, bottom=0.08, top=0.93, 
                     wspace=0.25, hspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# 绘图
plot_real_roc_with_ci(ax1, 'Internal', 'Internal Validation Set')
plot_real_roc_with_ci(ax2, 'External', 'External Validation Set')
plot_styled_cm_mean_only(ax3, 'Internal Test', 'Confusion Matrix (Internal)')
plot_styled_cm_mean_only(ax4, 'External Test', 'Confusion Matrix (External)')

# 标签
for ax, label in zip([ax1, ax2, ax3, ax4], ['a)', 'b)', 'c)', 'd)']):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, 
            fontsize=20, fontweight='bold', va='bottom', ha='right', family='Arial')

# 保存
plt.savefig('Figure2_MeanOnly_CustomColor.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Figure2_MeanOnly_CustomColor.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 2 生成完毕！\n特点：混淆矩阵仅展示均值，保留蓝紫色立体风格；ROC使用Nature配色且带真实阴影。")
