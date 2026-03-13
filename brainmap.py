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

# ========== Load BrainNetome atlas metadata ==========
def load_bna_atlas():
    # Read BrainnetomeAtlas_BNA_subregions.xlsx
    bna_df = pd.read_excel('BrainnetomeAtlas_BNA_subregions.xlsx')
    
    # Create a mapping dictionary from region code to lobe
    region_to_lobe = {}
    
    # Track the current lobe value
    current_lobe = None
    
    # Iterate rows and record region-to-lobe mappings
    for i, row in bna_df.iterrows():
        # Update current lobe when Lobe is not NaN
        if pd.notna(row['Lobe']):
            current_lobe = row['Lobe'].strip()  # Ensure whitespace is stripped
        
        # Extract region code when anatomical description exists
        if pd.notna(row['Anatomical and modified Cyto-architectonic descriptions']):
            desc = row['Anatomical and modified Cyto-architectonic descriptions']
            # Extract region code with regex (for example A8vl, A46)
            match = re.search(r'([A-Za-z0-9/]+)', desc)
            if match:
                region_code = match.group(1)
                # Link region code to current lobe
                if current_lobe:
                    region_to_lobe[region_code] = current_lobe
    
    # Print debug info to verify mappings
    print(f"Loaded lobe mappings for {len(region_to_lobe)} brain region codes")
    
    return region_to_lobe

# ========== Load and process SHAP data ==========
def process_shap_data(csv_file, region_to_lobe):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Keep only the first column (region) and second column (TRD SHAP value)
    df = df.iloc[:, :2]
    df.columns = ['region', 'value']
    
    # Filter out rows with value == 0
    df = df[df['value'] != 0]
    
    # Parse region string: extract prefix (alff/scfc) and suffix (L/R)
    df[['prefix', 'name', 'suffix']] = df['region'].str.split('_', n=2, expand=True)
    
    # Normalize prefix/suffix for consistent comparison
    df['prefix'] = df['prefix'].str.lower()
    df['suffix'] = df['suffix'].str.upper() if df['suffix'].notna().all() else df['suffix']
    
    # Define valid groups
    valid_prefixes = ['alff', 'scfc']
    valid_suffixes = ['L', 'R']
    
    # Keep only rows that satisfy filters
    df = df[df['prefix'].isin(valid_prefixes) & df['suffix'].isin(valid_suffixes)]
    
    # Compute absolute value
    df['abs_value'] = df['value'].abs()
    
    # Add lobe information
    def get_lobe(region_name):
        # Try direct match
        if region_name in region_to_lobe:
            return region_to_lobe[region_name]
        
        # Try first token match (for example A12 from A12/47o)
        parts = region_name.split('/')
        if parts[0] in region_to_lobe:
            return region_to_lobe[parts[0]]
        
        # Try other possible matching strategies
        for key in region_to_lobe.keys():
            if key in region_name or region_name in key:
                return region_to_lobe[key]
        
        return 'Unknown'
    
    df['lobe'] = df['name'].apply(get_lobe)
    
    # Preserve sign information
    df['color_type'] = np.where(df['value'] > 0, 'positive', 'negative')
    
    # Print debug info
    print("SHAP data processed, total rows:", len(df))
    
    return df

# ========== Define colors for each lobe ==========
def get_lobe_colors():
    # Define a distinct color for each lobe
    lobe_colors = {
        'Frontal Lobe': '#FF6347',  # tomato red
        'Temporal Lobe': '#4682B4',  # steel blue
        'Parietal Lobe': '#32CD32',  # lime green
        'Insular Lobe': '#9370DB',  # medium purple
        'Limbic Lobe': '#FFD700',  # gold
        'Occipital Lobe': '#8B4513',  # saddle brown
        'Subcortical Nuclei': '#FF69B4',  # hot pink
        'Unknown': '#808080'  # gray (for unknown regions)
    }
    return lobe_colors

# ========== Plot bar chart ==========
def create_bar_plots(df, prefix, ax, lobe_colors):
    """
    Create one integrated bar chart for both hemispheres with different markers.
    """
    # Select all data for this prefix (both hemispheres)
    subset = df[df['prefix'] == prefix.lower()]
    if not subset.empty:
        # Sort by absolute value and keep top 16 for display
        subset = subset.sort_values('abs_value', ascending=False)
  #      if len(subset) > 16:
   #         subset = subset.head(16)
        
        # Reset index for plotting
        subset = subset.reset_index(drop=True)
        
        # Get bar color for each entry
        bar_colors = [lobe_colors.get(lobe, lobe_colors['Unknown']) for lobe in subset['lobe']]
        
        # Build labels with hemisphere suffix
        bar_labels = [f"{row['name']}_{row['suffix']}" for _, row in subset.iterrows()]
        
        # Draw bar chart
        bars = ax.bar(bar_labels, subset['value'], color=bar_colors)
        
        # Add +/- markers to indicate sign
        for i, (value, bar) in enumerate(zip(subset['value'], bars)):
            if value > 0:
                marker_y = value + value * 0.05
                ax.text(i, marker_y, '+', color='black', ha='center', va='bottom', fontweight='bold', fontsize=8)
            else:
                marker_y = value - abs(value) * 0.05
                ax.text(i, marker_y, '-', color='black', ha='center', va='top', fontweight='bold', fontsize=8)
            
            # Add hemisphere marker
            suffix = subset.iloc[i]['suffix']
            if suffix == 'L':
                marker_color = 'blue'
                marker_symbol = 'o'  # circle = left hemisphere
            else:
                marker_color = 'red'
                marker_symbol = 's'  # square = right hemisphere
            
            # Place marker inside the bar region
            if value >= 0:
                y_pos = value * 0.5  # midpoint of positive bar
            else:
                y_pos = value * 0.5  # midpoint of negative bar
            
            ax.plot(i, y_pos, marker=marker_symbol, markersize=5, color=marker_color, markeredgecolor='black')
        
        # Rotate x labels for readability and reduce font size
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=8)
        
        # Add horizontal zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Set y-axis range based on data
        ax.set_ylim(subset['value'].min() * 1.1, subset['value'].max() * 1.1)
        
        # Collect lobes that appear in this panel
        displayed_lobes = set(subset['lobe'])
        
        # Add hemisphere legend near the lower-right corner
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, markeredgecolor='black', label='Left Hemisphere'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, markeredgecolor='black', label='Right Hemisphere')
        ]
        hemisphere_legend = ax.legend(handles=custom_lines, loc='lower right', bbox_to_anchor=(1.0, 0.02), 
                               fontsize=8, frameon=True, facecolor='white', edgecolor='lightgray', title="Hemisphere")
        hemisphere_legend.get_frame().set_alpha(0.8)
        
        # Create lobe legend elements
        lobe_legend_elements = [
            Line2D([0], [0], color=color, lw=4, 
                  path_effects=[path_effects.withStroke(linewidth=6, foreground=color+'80')],
                  label=lobe)
            for lobe, color in lobe_colors.items()
            if lobe in displayed_lobes and lobe != 'Unknown'  # Show only lobes that appear in this plot
        ]
        
        # Add lobe legend near lower-right, slightly higher to avoid overlap
        ax.add_artist(hemisphere_legend)  # Keep the first legend visible
        lobe_legend = ax.legend(handles=lobe_legend_elements, loc='lower right', bbox_to_anchor=(1.0, 0.25), 
                           fontsize=8, frameon=True, facecolor='white', edgecolor='lightgray',
                           title="Brain Lobes")
        lobe_legend.get_frame().set_alpha(0.8)
    else:
        ax.text(0.5, 0.5, f"No data for {prefix.upper()}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    # Use simplified labels; remove SHAP-values title
    # ax.set_title(f"{prefix.upper()} SHAP Values", fontsize=14)  # title removed
    ax.set_xlabel('Brain Region', fontsize=10)
    ax.set_ylabel('SHAP Value', fontsize=10)
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Keep figure content within bounds
    plt.tight_layout()
    
    # Return displayed lobes and SHAP value range
    return displayed_lobes, (subset['value'].min(), subset['value'].max())

# ========== Create colorbar legend ==========
def create_colorbar_legend(fig, pos, title, vmin, vmax):
    """Create a colorbar legend for brain maps at a specified position using actual value range."""
    # Create a linear colormap
    colors = ['navy', 'blue', 'lightblue', 'white', 'yellow', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # Create an axis at the specified position
    cbar_ax = fig.add_axes(pos)
    
    # Create normalization from actual min/max values
    norm = plt.Normalize(vmin, vmax)
    
    # Create colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                      cax=cbar_ax, orientation='horizontal')
    
    # Set ticks at key values (min, zero, max when applicable)
    if vmin < 0 and vmax > 0:
        # If range crosses zero, include zero tick
        ticks = [vmin, vmin/2, 0, vmax/2, vmax]
    else:
        # If all positive or all negative, use 5 evenly spaced ticks
        ticks = np.linspace(vmin, vmax, 5)
    
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{x:.2f}' for x in ticks])
    
    # Set title
    cbar_ax.set_title(title, fontsize=10)
    
    return cbar_ax

# ========== Create glass brain ==========
def create_glass_brain(nii_file, ax, display_mode, title=None):
    """Create a glass-brain plot on the given axis."""
    if not os.path.exists(nii_file):
        ax.text(0.5, 0.5, f"File not found: {nii_file}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return None, None
    
    try:
        # Load NIfTI file
        stat_map_img = image.load_img(nii_file)
        img_data = stat_map_img.get_fdata()
        
        # Compute an appropriate display range
        nonzero_data = img_data[np.abs(img_data) > 1e-10]
        if len(nonzero_data) > 0:
            display_max = max(abs(np.min(nonzero_data)), abs(np.max(nonzero_data)))
            display_max = np.ceil(display_max * 100) / 100
            vmin, vmax = -display_max, display_max
        else:
            vmin, vmax = -0.05, 0.05  # default range
        
        # Set colormap with enhanced contrast
        colors = ['navy', 'blue', 'lightblue', 'white', 'yellow', 'red', 'darkred']
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        
        # Create glass-brain plot
        display = plotting.plot_glass_brain(
            stat_map_img,
            display_mode=display_mode,
            threshold=1e-10,  # Use a small threshold to show more detail
            cmap=cmap,
            colorbar=False,  # Do not render colorbar inside this panel
            symmetric_cbar=True,
            vmin=vmin,
            vmax=vmax,
            annotate=False,
            black_bg=False,
            figure=plt.gcf(),
            axes=ax,
            plot_abs=False,
            alpha=0.9,  # Increase transparency
        )
        
        # Set title
       # if title:
        #    ax.set_title(title, fontsize=12, pad=0)
            
        # Return actual value range for colorbar usage
        return vmin, vmax
    
    except Exception as e:
        ax.text(0.5, 0.5, f"Error creating glass brain: {e}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return None, None

# ========== Create full visualization layout ==========
def create_combined_visualization():
    # Keep default font settings
    # plt.rcParams.update({
    #     'font.size': 16,
    #     'axes.titlesize': 18,
    #     'axes.labelsize': 16,
    #     'xtick.labelsize': 14,
    #     'ytick.labelsize': 14,
    #     'legend.fontsize': 14,
    #     'legend.title_fontsize': 16
    # })
    # Create figure object
    fig = plt.figure(figsize=(18, 14))  # Use a smaller overall brain-map size
    
    # Create main grid layout: ALFF on left, SFC on right
    gs_main = GridSpec(1, 2, figure=fig, 
                      width_ratios=[1, 1],
                      left=0.05, right=0.95, bottom=0.05, top=0.92,  # Adjust margins to keep all content inside bounds
                      wspace=0.2)  # Increase spacing between left and right sections
    
    # Load BNA atlas and color mappings
    region_to_lobe = load_bna_atlas()
    lobe_colors = get_lobe_colors()
    
    # Process SHAP data
    shap_df = process_shap_data('real_shap.csv', region_to_lobe)
    
    # ================ ALFF section (left) ================
    # Create ALFF grid: 3 rows for plots + 1 row for colorbar
    gs_alff = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_main[0], 
                                    hspace=0.3, height_ratios=[1, 1, 1, 0.2])  # Reserve 4th row for colorbar
    
    # ALFF title (kept optional at top-left)
    # alff_title_text = fig.text(0.25, 0.95, "ALFF", 
    #                         fontsize=20, fontweight='bold', ha='center', va='center',
    #                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', 
    #                                  boxstyle='round,pad=0.5', linewidth=0.5))
    
    # ===== ALFF bar-chart panel (top row) =====
    ax_alff_bar = fig.add_subplot(gs_alff[0])
    alff_lobes, alff_bar_range = create_bar_plots(shap_df, 'alff', ax_alff_bar, lobe_colors)
    # Add panel label a)
    ax_alff_bar.text(-0.05, 1.15, 'a)', transform=ax_alff_bar.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')
    
    # ===== ALFF left/right hemisphere views (middle row) =====
    gs_alff_lr = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_alff[1], 
                                       wspace=0.1)
    
    # ALFF left-hemisphere glass brain
    ax_alff_left_brain = fig.add_subplot(gs_alff_lr[0, 0])
    nii_path_alff = os.path.join('brainmap_python', 'ALFF.nii')
    alff_vmin, alff_vmax = create_glass_brain(nii_path_alff, ax_alff_left_brain, 'l', None)
    # Add panel label c)
    ax_alff_left_brain.text(-0.11, 0.95, 'c)', transform=ax_alff_left_brain.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')

    # ALFF right-hemisphere glass brain
    ax_alff_right_brain = fig.add_subplot(gs_alff_lr[0, 1])
    create_glass_brain(nii_path_alff, ax_alff_right_brain, 'r', 'Right Hemisphere View')
    
    # ===== ALFF Y/Z views (bottom row) =====
    gs_alff_yz = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_alff[2], 
                                       wspace=0.1)
    
    # ALFF Y view
    ax_alff_y = fig.add_subplot(gs_alff_yz[0, 0])
    create_glass_brain(nii_path_alff, ax_alff_y, 'y', 'Coronal View (Y)')
    
    # ALFF Z view
    ax_alff_z = fig.add_subplot(gs_alff_yz[0, 1])
    create_glass_brain(nii_path_alff, ax_alff_z, 'z', 'Axial View (Z)')
    
    # ALFF colorbar legend in 4th row using brain-map range
    create_colorbar_legend(fig, [0.08, 0.1, 0.34, 0.02], "ALFF SHAP Values", alff_vmin, alff_vmax)
    
    # ================ SFC section (right) ================
    # Create SFC grid: 3 rows for plots + 1 row for colorbar
    gs_sfc = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_main[1], 
                                   hspace=0.3, height_ratios=[1, 1, 1, 0.2])  # Reserve 4th row for colorbar
    
    # SFC title (kept optional at top-right)
    # sfc_title_text = fig.text(0.75, 0.95, "SFC", 
    #                        fontsize=20, fontweight='bold', ha='center', va='center',
    #                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', 
    #                                 boxstyle='round,pad=0.5', linewidth=0.5))
    
    # ===== SFC bar-chart panel (top row) =====
    ax_sfc_bar = fig.add_subplot(gs_sfc[0])
    sfc_lobes, sfc_bar_range = create_bar_plots(shap_df, 'scfc', ax_sfc_bar, lobe_colors)
    # Add panel label b)
    ax_sfc_bar.text(-0.05, 1.15, 'b)', transform=ax_sfc_bar.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')
    
    # ===== SFC left/right hemisphere views (middle row) =====
    gs_sfc_lr = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_sfc[1], 
                                      wspace=0.1)
    
    # SFC left-hemisphere glass brain
    ax_sfc_left_brain = fig.add_subplot(gs_sfc_lr[0, 0])
    nii_path_sfc = os.path.join('brainmap_python', 'SFC.nii')
    sfc_vmin, sfc_vmax = create_glass_brain(nii_path_sfc, ax_sfc_left_brain, 'l', None)
    # Add panel label d)
    ax_sfc_left_brain.text(-0.11, 0.95, 'd)', transform=ax_sfc_left_brain.transAxes, fontsize=25, fontweight='bold', va='top', ha='left')

    # SFC right-hemisphere glass brain
    ax_sfc_right_brain = fig.add_subplot(gs_sfc_lr[0, 1])
    create_glass_brain(nii_path_sfc, ax_sfc_right_brain, 'r', 'Right Hemisphere View')
    
    # ===== SFC Y/Z views (bottom row) =====
    gs_sfc_yz = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_sfc[2], 
                                      wspace=0.1)
    
    # SFC Y view
    ax_sfc_y = fig.add_subplot(gs_sfc_yz[0, 0])
    create_glass_brain(nii_path_sfc, ax_sfc_y, 'y', 'Coronal View (Y)')
    
    # SFC Z view
    ax_sfc_z = fig.add_subplot(gs_sfc_yz[0, 1])
    create_glass_brain(nii_path_sfc, ax_sfc_z, 'z', 'Axial View (Z)')
    
    # SFC colorbar legend in 4th row using brain-map range
    create_colorbar_legend(fig, [0.58, 0.1, 0.34, 0.02], "SFC SHAP Values", sfc_vmin, sfc_vmax)
    
    # Ensure output directory exists
    output_dir = 'visualization_results/brain_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output filename prefix
    output_prefix = os.path.join(output_dir, 'lobe_colors_final_layout')
    
    # Save high-quality vector and raster outputs
    plt.savefig(f'{output_prefix}.svg', dpi=800, facecolor='white')
    plt.savefig(f'{output_prefix}.pdf', dpi=800, facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=800, facecolor='white')
    
    # Save transparent-background version
    fig.patch.set_alpha(0.0)
    plt.savefig(f'{output_prefix}_transparent.png', dpi=800, transparent=True)
    
    print(f"Final brain visualization layout saved to '{output_dir}' folder:")
    print(f"- Vector formats: '{os.path.basename(output_prefix)}.svg' and '{os.path.basename(output_prefix)}.pdf'")
    print(f"- High-resolution raster: '{os.path.basename(output_prefix)}.png'")
    print(f"- Transparent background version: '{os.path.basename(output_prefix)}_transparent.png'")
    print("Features: ALFF on the left and SFC on the right; each side includes a bar chart (with dual legends at lower-right), left/right hemisphere views, Y/Z views, and colorbars based on actual value ranges.")

if __name__ == "__main__":
    create_combined_visualization() 
