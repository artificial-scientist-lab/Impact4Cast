import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Directory where npz files are stored
data_folder = "save_plot"
# Directory to save final full ROC plot
output_folder = "full_ROCs_single"
os.makedirs(output_folder, exist_ok=True)

# The set of y_diff values
y_diff_values = [1, 2, 3, 4, 5]

# Colors for the two fcNN variants
colors = {
    'single': 'blue',
    'mixed': 'red'
}

# Create a single figure with 5 columns of subplots, 1 row
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Add a main title to the entire figure
#fig.suptitle("Training from 2014 -> 2017 (Only fcNN Model Variants)", fontsize=22, y=1.08)

for i, y_diff in enumerate(y_diff_values):
    ax = axes[i]
    y2_eval = 2017 + y_diff

    # Patterns for the two fcNN variants:
    # single: IR=50 files
    single_pattern = f"fcNNsingle_eval_auc_curve_2017_{y2_eval}_*_50.npz"
    # mixed: IR=10 files
    mixed_pattern = f"fcNNmixed_eval_auc_curve_2017_{y2_eval}_*_10.npz"

    # Search for files
    single_files = glob.glob(os.path.join(data_folder, single_pattern))
    mixed_files = glob.glob(os.path.join(data_folder, mixed_pattern))

    any_data = False

    # Plot single variant if available
    if single_files:
        data = np.load(single_files[0])
        fpr = data['fpr']
        tpr = data['tpr']
        auc_score = data['auc_score']
        ax.plot(fpr, tpr, label=f"fcNN (IR=50) AUC={auc_score:.4f}", color=colors['single'])
        any_data = True

    # Plot mixed variant if available
    if mixed_files:
        data = np.load(mixed_files[0])
        fpr = data['fpr']
        tpr = data['tpr']
        auc_score = data['auc_score']
        ax.plot(fpr, tpr, label=f"fcNN (IR=10) AUC={auc_score:.4f}", color=colors['mixed'])
        any_data = True

    if any_data:
        # If at least one curve plotted, show the random line and legend
        ax.plot([0, 1], [0, 1], 'k--', label="Random AUC=0.5")
        ax.legend(loc="lower right")
    else:
        # If no data for this subplot
        ax.text(0.5, 0.5, "no data available",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Evaluation: 2017-{y2_eval}, IR=50")
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the single combined figure
output_filename = "full_ROC_grid_fcNN_variants.pdf"
plt.savefig(os.path.join(output_folder, output_filename), dpi=600)

plt.show()
plt.close()

print("Single figure with all fcNN variant subplots generated and shown.")
