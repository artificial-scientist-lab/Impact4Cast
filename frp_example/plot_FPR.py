import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# File names and corresponding imbalance ratios (IR)
files = ["solution_output_10.npy", "solution_output_50.npy", "solution_output_100.npy"]
labels = ["IR=10", "IR=50", "IR=100"]

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(28, 12))  # Increase the figure width
fig.subplots_adjust(wspace=2)  # Add whitespace between subplots

# Plot for single IR=10 (Left plot)
data = np.load("solution_output_10.npy")
y_true = data[:, 0]
y_scores = data[:, 1]

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

axes[0].plot(fpr, tpr, label=f'IR=10 (AUC = {roc_auc:.2f})', color='blue')

for fpr_value in [0.1, 0.3]:
    idx = np.argmin(np.abs(fpr - fpr_value))
    tpr_value = tpr[idx]
    print(f'For FPR={fpr_value}, TPR={tpr_value:.2f}')
    axes[0].plot([fpr_value, fpr_value], [0, tpr_value], linestyle='dotted', color='black', linewidth=3.5)  # Vertical line
    axes[0].plot([0, fpr_value], [tpr_value, tpr_value], linestyle='dotted', color='black', linewidth=3.5)  # Horizontal line
    axes[0].text(fpr_value + 0.02, tpr_value - 0.05, f'(FPR={fpr_value}, TPR={tpr_value:.2f})',
                 fontsize=36, color='black')  # Increased font size by 10

axes[0].set_xlabel('False Positive Rate', fontsize=40)  # Increased font size by 10
axes[0].set_ylabel('True Positive Rate', fontsize=40)  # Increased font size by 10
axes[0].set_title('ROC Curve with Thresholds (IR=10)', fontsize=44)  # Increased font size by 10
axes[0].tick_params(axis='both', which='major', labelsize=36)  # Increased font size by 10
axes[0].grid()
axes[0].legend(loc='lower right', fontsize=36)  # Increased font size by 10

# Plot for all IRs (Right plot)
for file, label in zip(files, labels):
    data = np.load(file)
    y_true = data[:, 0]
    y_scores = data[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})', linewidth=3)  # Increase line width


axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
axes[1].set_xlabel('False Positive Rate', fontsize=40)  # Increased font size by 10
axes[1].set_ylabel('True Positive Rate', fontsize=40)  # Increased font size by 10
axes[1].set_title('ROC Curve for IR=10, 50, 100', fontsize=44)  # Increased font size by 10
axes[1].tick_params(axis='both', which='major', labelsize=36)  # Increased font size by 10
axes[1].grid()
axes[1].legend(loc='lower right', fontsize=36)  # Increased font size by 10

# Adjust layout and save the combined plot
plt.tight_layout()
plt.savefig('roc_curve_combined_highres.png', dpi=300)
plt.show()
