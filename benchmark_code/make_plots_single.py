import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Directory where npz files are stored
data_folder = "save_plot"
# Directory to save final full ROC plot
output_folder = "full_ROCs_single"
os.makedirs(output_folder, exist_ok=True)

model_colors = {
    'fcNN': 'blue',
    'tree': 'red',
    'xgboost': 'green',
    'transformer': 'orange'
}

# The set of models to include
models = ['fcNN', 'transformer', 'tree', 'xgboost']

# y_diff from 1 to 5
y_diff_values = [1, 2, 3, 4, 5]
# IR values
IR_values = [10, 50]

# Create a single figure with 2 rows and 5 columns of subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# Add a main title to the entire figure
fig.suptitle("Training from 2014 -> 2017", fontsize=22, y=0.96)

for i, y_diff in enumerate(y_diff_values):
    y1_eval = 2017
    y2_eval = 2017 + y_diff
    
    for j, IR in enumerate(IR_values):
        ax = axes[j, i]  # j is row index (0 for IR=10, 1 for IR=50), i is column index for y_diff

        any_model_plotted = False  # To check if we plotted any model
        
        # Attempt to plot models
        for model in models:
            if model == 'fcNN':
                pattern = f"{model}single_eval_auc_curve_2017_{y2_eval}_*_{IR}.npz"
            else:
                # For 'transformer', 'tree', 'xgboost'
                pattern = f"{model}single_eval_auc_curve_2017_2017_*_{IR}_2017_{y2_eval}.npz"

            search_pattern = os.path.join(data_folder, pattern)
            files = glob.glob(search_pattern)

            if not files:
                continue

            npz_file = files[0]
            data = np.load(npz_file)
            fpr = data['fpr']
            tpr = data['tpr']
            auc_score = data['auc_score']
            
            
            if model=='fcNN':
                write_model='fcNN'
            if model=='tree':
                write_model='Forest'
            if model=='xgboost':
                write_model='XGBoost'
            if model=='transformer':
                write_model='Transformer'              
        
            ax.plot(fpr, tpr, label=f"{write_model} AUC={auc_score:.4f}", color=model_colors[model])
            any_model_plotted = True

        if any_model_plotted:
            # If we found at least one model, then plot the random diagonal
            ax.plot([0, 1], [0, 1], 'k--', label="Random AUC=0.5")
            ax.legend(loc="lower right")
        else:
            # If no model was plotted on this subplot, add a message
            ax.text(0.5, 0.5, "not enough data for evaluation", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"Evaluation: 2017-{y2_eval}, IR={IR}")
        ax.grid(True)

# Adjust layout so things fit nicely, and leave space for suptitle
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the single combined figure
output_filename = "full_ROC_grid_single.pdf"
plt.savefig(os.path.join(output_folder, output_filename), dpi=600)

plt.show()
plt.close()

print("Single figure with all subplots generated and shown.")
