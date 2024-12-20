import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Define the model colors to ensure consistency
model_colors = {
    'fcNN': 'blue',
    'tree': 'red',
    'xgboost': 'green',
    'transformer': 'orange'
}

# (y1_train, y2_train) pairs to consider based on conditions (already given)
train_pairs = [
    (y1, y2)
    for y1 in range(2013, 2020)
    for y2 in range(2017, 2022)
    if y2 - y1 in [2, 3, 4]
]

IR_values = [10, 50]

data_folder = "save_plot"
plot_folder = "full_ROCs"
os.makedirs(plot_folder, exist_ok=True)

# Helper function to parse filename and extract model, y1, y2, IR
def parse_filename(filename):
    base = os.path.basename(filename)
    base = base.replace('.npz', '')
    parts = base.split('_')
    
    # Find 'eval' as an anchor
    try:
        eval_idx = parts.index('eval')
    except ValueError:
        return None, None, None, None
    
    # Model name is everything before 'eval'
    model_name = '_'.join(parts[:eval_idx])
    
    # Check for minimum length
    if len(parts) < eval_idx + 7:
        return None, None, None, None

    # After eval_auc_curve come y1, y2 and IR
    try:
        y1_train = int(parts[eval_idx+3])
        y2_train = int(parts[eval_idx+4])
        IR = int(parts[-1])
    except ValueError:
        return None, None, None, None
    
    return model_name, y1_train, y2_train, IR

print("Collecting files...")

# Collect all .npz files
all_files = glob.glob(os.path.join(data_folder, "*.npz"))



# Dictionary to store files by (y1, y2, IR)
files_dict = {}
for f in all_files:
    if 'single' in f:
        # Ignore files containing "single"
        continue
    model_name, y1_t, y2_t, IR = parse_filename(f)
    if model_name is None:
        continue
    # Ensure model_name is in our known set
    if model_name not in ['fcNN', 'tree', 'xgboost', 'transformer']:
        continue
    
    key = (y1_t, y2_t, IR)
    if key not in files_dict:
        files_dict[key] = {}
    if model_name not in files_dict[key]:
        files_dict[key][model_name] = []
    files_dict[key][model_name].append(f)

print(f"Collected files for {len(files_dict)} (y1,y2,IR) combinations.")

# We want two figures: one for IR=10 and one for IR=50
# Rows: delta_train in [2,3,4]
# Cols: delta_eval in [1,2,3,4,5]
delta_train_values = [2, 3, 4]
delta_eval_values = [1, 2, 3, 4, 5]



print("Starting plotting...")

for IR in IR_values:
    fig, axes = plt.subplots(len(delta_train_values), len(delta_eval_values), figsize=(20, 12))
    #fig.suptitle(f"Various Training and Evaluation Intervals (IR={IR})", fontsize=24, y=0.96)

    for row, dt in enumerate(delta_train_values):
        for col, de in enumerate(delta_eval_values):
            ax = axes[row, col]

            # Compute y1, y2 from dt and de:
            # delta_eval = 2022 - y2 => y2 = 2022 - delta_eval = 2022 - de
            y2 = 2022 - de
            # delta_train = y2 - y1 => y1 = y2 - dt
            y1 = y2 - dt

            key = (y1, y2, IR)
            if key not in files_dict or not files_dict[key]:
                # No data for this combination
                ax.text(0.5, 0.5, "not enough data for evaluation", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Train: {y1}-{y2}, Eval: {y2}-2022")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.grid(True)
                continue

            model_files = files_dict[key]

            # Plot each model's ROC curve (take first file if multiple)
            any_plotted = False
            for model_name in model_files:
                f = model_files[model_name][0]
                data = np.load(f)
                fpr = data['fpr']
                tpr = data['tpr']
                auc_score = data['auc_score']
                
                if model_name=='fcNN':
                    write_model='fcNN'
                if model_name=='tree':
                    write_model='Forest'
                if model_name=='xgboost':
                    write_model='XGBoost'
                if model_name=='transformer':
                    write_model='Transformer'                    
                ax.plot(fpr, tpr, color=model_colors.get(model_name, 'black'),
                        label=f"{write_model} AUC={auc_score:.4f}")
                any_plotted = True

            if any_plotted:
                # Plot the random line
                ax.plot([0, 1], [0, 1], 'k--', label="Random AUC=0.5")
                ax.legend(loc="lower right")
            else:
                # If no model plotted (though files_dict said otherwise, just in case)
                ax.text(0.5, 0.5, "not enough data for evaluation", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)

            ax.set_title(f"Train: {y1}-{y2}, Eval: {y2}-2022")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_filename = f"comparison_roc_grid_IR{IR}.pdf"
    save_path = os.path.join(plot_folder, plot_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined figure for IR={IR} to {save_path}")

print("All plots have been generated.")
