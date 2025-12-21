import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from nilearn import image, plotting
from pathlib import Path
import re
import matplotlib.patheffects as path_effects

# ========== 读取BrainNetome Atlas信息 ==========
def load_bna_atlas():
    # 读取BrainnetomeAtlas_BNA_subregions.xlsx文件
    bna_df = pd.read_excel('BrainnetomeAtlas_BNA_subregions.xlsx')
    
    # 创建空字典来存储脑区-Lobe映射
    region_to_lobe = {}
    
    # 跟踪当前的Lobe值
    current_lobe = None
    
    # 遍历每一行，记录区域与Lobe的对应关系
    for i, row in bna_df.iterrows():
        # 如果Lobe不是NaN，就更新当前Lobe
        if pd.notna(row['Lobe']):
            current_lobe = row['Lobe'].strip()  # 确保移除任何空格
        
        # 如果有解剖描述，提取区域代码
        if pd.notna(row['Anatomical and modified Cyto-architectonic descriptions']):
            desc = row['Anatomical and modified Cyto-architectonic descriptions']
            # 使用正则表达式提取区域代码（如A8vl, A46等）
            match = re.search(r'([A-Za-z0-9/]+)', desc)
            if match:
                region_code = match.group(1)
                # 将区域代码与当前Lobe关联
                if current_lobe:
                    region_to_lobe[region_code] = current_lobe
    
    # 输出一些调试信息，帮助验证映射
    print(f"已加载{len(region_to_lobe)}个脑区代码的Lobe映射")
    
    return region_to_lobe

# ========== 读取和处理SHAP数据 ==========
def process_shap_data(csv_file, region_to_lobe):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 只保留第一列(region)和第二列(TRD的SHAP值)
    df = df.iloc[:, :2]
    df.columns = ['region', 'value']
    
    # 过滤掉值为0的行
    df = df[df['value'] != 0]
    
    # 解析区域信息：提取前缀(alff/scfc)和后缀(L/R)
    df[['prefix', 'name', 'suffix']] = df['region'].str.split('_', n=2, expand=True)
    
    # 确保前缀和后缀转为小写以便统一比较
    df['prefix'] = df['prefix'].str.lower()
    df['suffix'] = df['suffix'].str.upper() if df['suffix'].notna().all() else df['suffix']
    
    # 过滤有效分组
    valid_prefixes = ['alff', 'scfc']
    valid_suffixes = ['L', 'R']
    
    # 只保留符合条件的行
    df = df[df['prefix'].isin(valid_prefixes) & df['suffix'].isin(valid_suffixes)]
    
    # 计算绝对值
    df['abs_value'] = df['value'].abs()
    
    # 添加Lobe信息
    def get_lobe(region_name):
        # 尝试直接匹配
        if region_name in region_to_lobe:
            return region_to_lobe[region_name]
        
        # 尝试匹配区域名的第一部分（例如从A12/47o匹配A12）
        parts = region_name.split('/')
        if parts[0] in region_to_lobe:
            return region_to_lobe[parts[0]]
        
        # 尝试其他可能的匹配方式
        for key in region_to_lobe.keys():
            if key in region_name or region_name in key:
                return region_to_lobe[key]
        
        return 'Unknown'
    
    df['lobe'] = df['name'].apply(get_lobe)
    
    # 保留正负值信息
    df['color_type'] = np.where(df['value'] > 0, 'positive', 'negative')
    
    # 输出调试信息
    print("SHAP数据已处理，共有", len(df), "行")
    
    return df

# ========== 定义Lobe对应的颜色 ==========
def get_lobe_colors():
    # 为每个Lobe定义一个独特的颜色
    lobe_colors = {
        'Frontal Lobe': '#FF6347',  # 番茄红
        'Temporal Lobe': '#4682B4',  # 钢蓝
        'Parietal Lobe': '#32CD32',  # 酸橙绿
        'Insular Lobe': '#9370DB',  # 中紫色
        'Limbic Lobe': '#FFD700',  # 金色
        'Occipital Lobe': '#8B4513',  # 马鞍棕色
        'Subcortical Nuclei': '#FF69B4',  # 热粉红
        'Unknown': '#808080'  # 灰色（用于未知区域）
    }
    return lobe_colors

# ========== 绘制柱状图 ==========
def create_bar_plots(df, prefix, ax, lobe_colors):
    """
    创建单个柱状图，将左右脑数据集成到一个图中，并使用不同标记区分左右脑
    """
    # 选择当前前缀的所有数据（包括左右脑）
    subset = df[df['prefix'] == prefix.lower()]
    if not subset.empty:
        # 按绝对值排序，取前16个显示（左右脑各8个）
        subset = subset.sort_values('abs_value', ascending=False)
  #      if len(subset) > 16:
   #         subset = subset.head(16)
        
        # 为了柱状图显示，创建一个索引
        subset = subset.reset_index(drop=True)
        
        # 获取每个柱子对应的颜色
        bar_colors = [lobe_colors.get(lobe, lobe_colors['Unknown']) for lobe in subset['lobe']]
        
        # 为柱子添加左右脑的标识
        bar_labels = [f"{row['name']}_{row['suffix']}" for _, row in subset.iterrows()]
        
        # 绘制柱状图
        bars = ax.bar(bar_labels, subset['value'], color=bar_colors)
        
        # 在正值柱子上方和负值柱子下方添加小标记表示其正负性
        for i, (value, bar) in enumerate(zip(subset['value'], bars)):
            if value > 0:
                marker_y = value + value * 0.05
                ax.text(i, marker_y, '+', color='black', ha='center', va='bottom', fontweight='bold', fontsize=8)
            else:
                marker_y = value - abs(value) * 0.05
                ax.text(i, marker_y, '-', color='black', ha='center', va='top', fontweight='bold', fontsize=8)
            
            # 添加左右脑标记
            suffix = subset.iloc[i]['suffix']
            if suffix == 'L':
                marker_color = 'blue'
                marker_symbol = 'o'  # 圆形表示左脑
            else:
                marker_color = 'red'
                marker_symbol = 's'  # 方形表示右脑
            
            # 在柱子上方或下方添加标记
            if value >= 0:
                y_pos = value * 0.5  # 正值柱子的中点
            else:
                y_pos = value * 0.5  # 负值柱子的中点
            
            ax.plot(i, y_pos, marker=marker_symbol, markersize=5, color=marker_color, markeredgecolor='black')
        
        # 旋转x轴标签，使其更易读，减小字体
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        
        # 添加水平线表示零值
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 设置y轴范围，不再要求正负值域相同
        ax.set_ylim(subset['value'].min() * 1.1, subset['value'].max() * 1.1)
        
        # 收集此图中显示的脑叶集合
        displayed_lobes = set(subset['lobe'])
        
        # 添加左右脑标记的图例（在右下角靠下的位置）
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, markeredgecolor='black', label='Left Hemisphere'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, markeredgecolor='black', label='Right Hemisphere')
        ]
        hemisphere_legend = ax.legend(handles=custom_lines, loc='lower right', bbox_to_anchor=(1.0, 0.02), 
                               fontsize=8, frameon=True, facecolor='white', edgecolor='lightgray', title="Hemisphere")
        hemisphere_legend.get_frame().set_alpha(0.8)
        
        # 创建Lobe图例元素
        lobe_legend_elements = [
            Line2D([0], [0], color=color, lw=4, 
                  path_effects=[path_effects.withStroke(linewidth=6, foreground=color+'80')],
                  label=lobe)
            for lobe, color in lobe_colors.items()
            if lobe in displayed_lobes and lobe != 'Unknown'  # 只显示图中实际出现的脑区
        ]
        
        # 添加Lobe图例，也放在右下角但位置稍高，避免重叠
        ax.add_artist(hemisphere_legend)  # 确保第一个图例不被覆盖
        lobe_legend = ax.legend(handles=lobe_legend_elements, loc='lower right', bbox_to_anchor=(1.0, 0.25), 
                           fontsize=8, frameon=True, facecolor='white', edgecolor='lightgray',
                           title="Brain Lobes")
        lobe_legend.get_frame().set_alpha(0.8)
    else:
        ax.text(0.5, 0.5, f"No data for {prefix.upper()}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    # 设置简化的标题和标签，去掉"SHAP Values"
    # ax.set_title(f"{prefix.upper()} SHAP Values", fontsize=14)  # 移除标题
    ax.set_xlabel('Brain Region', fontsize=10)
    ax.set_ylabel('SHAP Value', fontsize=10)
    
    # 移除顶部和右侧的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 确保图形完全在图框内
    plt.tight_layout()
    
    # 返回此子图中显示的脑叶集合和SHAP值的范围
    return displayed_lobes, (subset['value'].min(), subset['value'].max())

# ========== 创建颜色条图例 ==========
def create_colorbar_legend(fig, pos, title, vmin, vmax):
    """为脑图创建颜色条图例，放置在指定位置，使用实际值域"""
    # 创建一个线性颜色映射
    colors = ['navy', 'blue', 'lightblue', 'white', 'yellow', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # 在指定位置创建一个轴
    cbar_ax = fig.add_axes(pos)
    
    # 根据实际脑图的最大最小值创建标准化对象
    norm = plt.Normalize(vmin, vmax)
    
    # 创建颜色条
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                      cax=cbar_ax, orientation='horizontal')
    
    # 设置刻度，在最小值、0和最大值处设置刻度
    if vmin < 0 and vmax > 0:
        # 如果跨越0，则在最小值、0和最大值设置刻度
        ticks = [vmin, vmin/2, 0, vmax/2, vmax]
    else:
        # 如果全为正或全为负，则均匀分布5个刻度点
        ticks = np.linspace(vmin, vmax, 5)
    
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{x:.2f}' for x in ticks])
    
    # 设置标题
    cbar_ax.set_title(title, fontsize=10)
    
    return cbar_ax

# ========== 创建玻璃脑图 ==========
def create_glass_brain(nii_file, ax, display_mode, title=None):
    """在指定的轴上创建玻璃脑图"""
    if not os.path.exists(nii_file):
        ax.text(0.5, 0.5, f"File not found: {nii_file}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return None, None
    
    try:
        # 加载nii文件
        stat_map_img = image.load_img(nii_file)
        img_data = stat_map_img.get_fdata()
        
        # 计算合适的显示范围
        nonzero_data = img_data[np.abs(img_data) > 1e-10]
        if len(nonzero_data) > 0:
            display_max = max(abs(np.min(nonzero_data)), abs(np.max(nonzero_data)))
            display_max = np.ceil(display_max * 100) / 100
            vmin, vmax = -display_max, display_max
        else:
            vmin, vmax = -0.05, 0.05  # 默认值
        
        # 设置colorbar颜色 - 增强对比度
        colors = ['navy', 'blue', 'lightblue', 'white', 'yellow', 'red', 'darkred']
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        
        # 创建玻璃脑图
        display = plotting.plot_glass_brain(
            stat_map_img,
            display_mode=display_mode,
            threshold=1e-10,  # 使用小阈值以显示更多细节
            cmap=cmap,
            colorbar=False,  # 不在视图中显示colorbar
            symmetric_cbar=True,
            vmin=vmin,
            vmax=vmax,
            annotate=False,
            black_bg=False,
            figure=plt.gcf(),
            axes=ax,
            plot_abs=False,
            alpha=0.9,  # 增加透明度
        )
        
        # 设置标题
       # if title:
        #    ax.set_title(title, fontsize=12, pad=0)
            
        # 返回实际使用的值域，供颜色条使用
        return vmin, vmax
    
    except Exception as e:
        ax.text(0.5, 0.5, f"Error creating glass brain: {e}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return None, None

# ========== 创建完整可视化 ==========
def create_combined_visualization():
    # 删除整体字体调大设置，恢复原始字体
    # plt.rcParams.update({
    #     'font.size': 16,
    #     'axes.titlesize': 18,
    #     'axes.labelsize': 16,
    #     'xtick.labelsize': 14,
    #     'ytick.labelsize': 14,
    #     'legend.fontsize': 14,
    #     'legend.title_fontsize': 16
    # })
    # 创建Figure对象
    fig = plt.figure(figsize=(18, 14))  # 让整体脑图变小
    
    # 创建主网格布局 - 左侧ALFF，右侧SFC
    gs_main = GridSpec(1, 2, figure=fig, 
                      width_ratios=[1, 1],
                      left=0.05, right=0.95, bottom=0.05, top=0.92,  # 调整边距，确保内容在边界内
                      wspace=0.2)  # 增加两部分之间的间距
    
    # 加载BNA Atlas和颜色映射
    region_to_lobe = load_bna_atlas()
    lobe_colors = get_lobe_colors()
    
    # 处理SHAP数据
    shap_df = process_shap_data('real_shap.csv', region_to_lobe)
    
    # ================ ALFF 部分（左侧） ================
    # 创建ALFF的网格布局，分为3行1列，第4行用于颜色条
    gs_alff = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_main[0], 
                                    hspace=0.3, height_ratios=[1, 1, 1, 0.2])  # 添加第4行用于颜色条
    
    # ALFF标题 - 放在左侧顶部
    # alff_title_text = fig.text(0.25, 0.95, "ALFF", 
    #                         fontsize=20, fontweight='bold', ha='center', va='center',
    #                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', 
    #                                  boxstyle='round,pad=0.5', linewidth=0.5))
    
    # ===== ALFF柱状图区域（上排）=====
    ax_alff_bar = fig.add_subplot(gs_alff[0])
    alff_lobes, alff_bar_range = create_bar_plots(shap_df, 'alff', ax_alff_bar, lobe_colors)
    # 添加a)标注
    ax_alff_bar.text(-0.05, 1.15, 'a)', transform=ax_alff_bar.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')
    
    # ===== ALFF左右脑视图（中排）=====
    gs_alff_lr = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_alff[1], 
                                       wspace=0.1)
    
    # ALFF左脑玻璃脑图
    ax_alff_left_brain = fig.add_subplot(gs_alff_lr[0, 0])
    nii_path_alff = os.path.join('brainmap_python', 'ALFF.nii')
    alff_vmin, alff_vmax = create_glass_brain(nii_path_alff, ax_alff_left_brain, 'l', None)
    # 添加c)标注
    ax_alff_left_brain.text(-0.11, 0.95, 'c)', transform=ax_alff_left_brain.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')

    # ALFF右脑玻璃脑图
    ax_alff_right_brain = fig.add_subplot(gs_alff_lr[0, 1])
    create_glass_brain(nii_path_alff, ax_alff_right_brain, 'r', 'Right Hemisphere View')
    
    # ===== ALFF Y和Z视图（下排）=====
    gs_alff_yz = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_alff[2], 
                                       wspace=0.1)
    
    # ALFF Y视图
    ax_alff_y = fig.add_subplot(gs_alff_yz[0, 0])
    create_glass_brain(nii_path_alff, ax_alff_y, 'y', 'Coronal View (Y)')
    
    # ALFF Z视图
    ax_alff_z = fig.add_subplot(gs_alff_yz[0, 1])
    create_glass_brain(nii_path_alff, ax_alff_z, 'z', 'Axial View (Z)')
    
    # ALFF颜色条图例 - 放在第4行位置，对应脑图的值域
    create_colorbar_legend(fig, [0.08, 0.1, 0.34, 0.02], "ALFF SHAP Values", alff_vmin, alff_vmax)
    
    # ================ SFC 部分（右侧） ================
    # 创建SFC的网格布局，分为3行1列，第4行用于颜色条
    gs_sfc = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_main[1], 
                                   hspace=0.3, height_ratios=[1, 1, 1, 0.2])  # 添加第4行用于颜色条
    
    # SFC标题 - 放在右侧顶部
    # sfc_title_text = fig.text(0.75, 0.95, "SFC", 
    #                        fontsize=20, fontweight='bold', ha='center', va='center',
    #                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', 
    #                                 boxstyle='round,pad=0.5', linewidth=0.5))
    
    # ===== SFC柱状图区域（上排）=====
    ax_sfc_bar = fig.add_subplot(gs_sfc[0])
    sfc_lobes, sfc_bar_range = create_bar_plots(shap_df, 'scfc', ax_sfc_bar, lobe_colors)
    # 添加b)标注
    ax_sfc_bar.text(-0.05, 1.15, 'b)', transform=ax_sfc_bar.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')
    
    # ===== SFC左右脑视图（中排）=====
    gs_sfc_lr = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_sfc[1], 
                                      wspace=0.1)
    
    # SFC左脑玻璃脑图
    ax_sfc_left_brain = fig.add_subplot(gs_sfc_lr[0, 0])
    nii_path_sfc = os.path.join('brainmap_python', 'SFC.nii')
    sfc_vmin, sfc_vmax = create_glass_brain(nii_path_sfc, ax_sfc_left_brain, 'l', None)
    # 添加d)标注
    ax_sfc_left_brain.text(-0.11, 0.95, 'd)', transform=ax_sfc_left_brain.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')

    # SFC右脑玻璃脑图
    ax_sfc_right_brain = fig.add_subplot(gs_sfc_lr[0, 1])
    create_glass_brain(nii_path_sfc, ax_sfc_right_brain, 'r', 'Right Hemisphere View')
    
    # ===== SFC Y和Z视图（下排）=====
    gs_sfc_yz = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_sfc[2], 
                                      wspace=0.1)
    
    # SFC Y视图
    ax_sfc_y = fig.add_subplot(gs_sfc_yz[0, 0])
    create_glass_brain(nii_path_sfc, ax_sfc_y, 'y', 'Coronal View (Y)')
    
    # SFC Z视图
    ax_sfc_z = fig.add_subplot(gs_sfc_yz[0, 1])
    create_glass_brain(nii_path_sfc, ax_sfc_z, 'z', 'Axial View (Z)')
    
    # SFC颜色条图例 - 放在第4行位置，对应脑图的值域
    create_colorbar_legend(fig, [0.58, 0.1, 0.34, 0.02], "SFC SHAP Values", sfc_vmin, sfc_vmax)
    
    # 确保输出目录存在
    output_dir = 'visualization_results/brain_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义输出文件名前缀
    output_prefix = os.path.join(output_dir, 'lobe_colors_final_layout')
    
    # 保存为高质量的矢量图和位图
    plt.savefig(f'{output_prefix}.svg', dpi=800, facecolor='white')
    plt.savefig(f'{output_prefix}.pdf', dpi=800, facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=800, facecolor='white')
    
    # 创建透明背景版本
    fig.patch.set_alpha(0.0)
    plt.savefig(f'{output_prefix}_transparent.png', dpi=800, transparent=True)
    
    print(f"最终布局的脑图可视化已保存至 '{output_dir}' 文件夹:")
    print(f"- 矢量格式: '{os.path.basename(output_prefix)}.svg'和'{os.path.basename(output_prefix)}.pdf'")
    print(f"- 高分辨率位图: '{os.path.basename(output_prefix)}.png'")
    print(f"- 透明背景版本: '{os.path.basename(output_prefix)}_transparent.png'")
    print("特点: 左侧ALFF右侧SFC，每侧包含柱状图(带有右下角双图例)、左右脑视图和Y/Z视图，脑图颜色条图例使用实际值域")

if __name__ == "__main__":
    create_combined_visualization() 
