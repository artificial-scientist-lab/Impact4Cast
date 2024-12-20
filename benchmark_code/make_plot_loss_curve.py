import os
import numpy as np
import matplotlib.pyplot as plt

# Define the directory and file name
base_dir = "save_plot"  # Directory name
file_name = "fcNN_loss_curve_2017_2020_600_0.0001_2048_500_10.npz"

# Construct full path using os.path
file_path = os.path.join(base_dir, file_name)

# Load the npz file
if os.path.exists(file_path):
    data = np.load(file_path)
    loss_train = data['loss_train']
    loss_test = data['loss_test']
else:
    raise FileNotFoundError(f"Error: File '{file_path}' not found.")

# Main plot
plt.figure(figsize=(10, 6))
plt.plot(loss_train, label='Train Loss', color='blue')
plt.plot(loss_test, label='Test Loss', color='orange')
plt.xlabel("Epoch", fontsize=24)
plt.ylabel("Loss", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Training and Test Loss Over Epochs", fontsize=28)
plt.legend(fontsize=20)

# Inset plot: Zoomed in from episode 1000 to the end
inset_start = 1000
ax_inset = plt.axes([0.30, 0.30, 0.55, 0.40])  # Position of inset: [x, y, width, height]
ax_inset.plot(range(inset_start, len(loss_train)), loss_train[inset_start:], label='Train Loss', color='blue')
ax_inset.plot(range(inset_start, len(loss_test)), loss_test[inset_start:], label='Test Loss', color='orange')
ax_inset.set_title(f"Zoomed In, Start at Epoch {inset_start}", fontsize=14)
ax_inset.set_xlabel("Epoch", fontsize=16)
ax_inset.set_ylabel("Loss", fontsize=16)
ax_inset.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
# Save the figure
output_dir = "save_plot_output"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "loss_curve_with_inset.png")
plt.savefig(save_path, dpi=300)

plt.show()
