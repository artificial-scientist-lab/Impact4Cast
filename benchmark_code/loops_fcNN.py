import os
import pickle
import gzip
import copy
import random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, date
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for cluster
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def get_predictions(model, data, solution, eval_batch_size, log_file):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        data_batches = torch.split(data, eval_batch_size)
        total_batches = len(data_batches)
        
        all_predictions = []
        start_time = time.time()
        for i, batch in enumerate(data_batches, start=1):
            batch_start_time = time.time()
            batch_preds = model(batch).squeeze().cpu().numpy()
            all_predictions.append(batch_preds)
            batch_time = time.time() - batch_start_time
            print_cluster(f"Processed batch {i}/{total_batches} in {batch_time:.2f} seconds", log_file)

        predictions = np.concatenate(all_predictions)
        
    true_labels = solution.cpu().numpy()  # Move labels to CPU
    total_time = time.time() - start_time
    print_cluster(f"Total prediction time: {total_time:.2f} seconds", log_file)
    return predictions, true_labels


def plot_auc_roc(true_labels, predictions, save_file, label="Train"):
    # Calculate the AUC-ROC score and ROC curve
    auc_score = roc_auc_score(true_labels, predictions)
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{label} AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', label=f"Random AUC={0.5}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve -- {label}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(save_file, dpi=300)
    plt.close()

    # Save data used to produce this figure, including predictions and ground truth
    data_file = save_file.replace('.png', '.npz')
    np.savez(
        data_file,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc_score=auc_score,
        true_labels=true_labels,
        predictions=predictions
    )
    return auc_score

def plot_loss_curve(loss_train, loss_test, save_file, label="year1-year2"):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_train, label=f'Train: {label}')
    plt.plot(loss_test, label=f'Test: {label}')
    plt.title(f"Loss Over Epochs: {label}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_file, dpi=300)
    plt.close()

    # Save data used to produce this figure
    data_file = save_file.replace('.png', '.npz')
    np.savez(data_file, loss_train=loss_train, loss_test=loss_test)


class ff_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ff_network, self).__init__()
        
        act = nn.ReLU()

        self.semnet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act,
            nn.Linear(hidden_size, hidden_size),
            act,
            nn.Linear(hidden_size, hidden_size),
            act,
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        res = self.semnet(x)
        return res

# Hyperparameters that we also use in naming
batch_size = 2048
lr_enc = 1e-4
hidden_size = 600
patience = 500

# Create folders if needed
neuralNet_folder = "save_neuralNet"
plot_folder = "save_plot"
os.makedirs(neuralNet_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

train_data_folder = "data_for_train"
eval_data_folder = "data_for_eval"
os.makedirs(train_data_folder, exist_ok=True)
os.makedirs(eval_data_folder, exist_ok=True)

rnd_seed = 42
random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)

# We define the loops as requested:
train_year_spans = [2, 3, 4]   # (y2_train - y1_train)
eval_year_spans = [1, 2, 3, 4, 5]  # (y2_eval - y1_eval)
IR_list = [10, 50]

fixed_y2_eval = 2022

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_cluster(print_str, log_file):
    print(print_str)
    with open(log_file, "a") as logfile:
        logfile.write(print_str + "\n")


for t_span in train_year_spans:
    for e_span in eval_year_spans:
        # Compute eval years
        y2_eval = fixed_y2_eval
        y1_eval = y2_eval - e_span

        # Compute train years
        # given: y2_train = y1_eval
        y2_train = y1_eval
        y1_train = y2_train - t_span

        # Define the log file name based on parameters
        for IR in IR_list:
            # Construct a file name prefix for all files

            file_name_prefix = f"{y1_train}_{y2_train}_{hidden_size}_{lr_enc}_{batch_size}_{patience}_{IR}"
            log_file = f'log_fcNN_{file_name_prefix}.txt'
            
            
            # Construct file paths
            if y1_train==2013:
                train_file = os.path.join(train_data_folder, f"train_data_{y1_train}_{y2_train}_IR{IR}_star.parquet")
                eval_file = os.path.join(eval_data_folder, f"eval_data_{y1_eval}_{y2_eval}_IR{IR}_star.parquet")
            else:
                train_file = os.path.join(train_data_folder, f"train_data_{y1_train}_{y2_train}_IR{IR}.parquet")
                eval_file = os.path.join(eval_data_folder, f"eval_data_{y1_eval}_{y2_eval}_IR{IR}.parquet")

            # Check if files exist before proceeding
            if not os.path.exists(train_file) or not os.path.exists(eval_file):
                print_cluster(f"Skipping (Train: {y1_train}-{y2_train}, Eval: {y1_eval}-{y2_eval}, IR={IR}) because files do not exist.", log_file)
                continue

            print_cluster(f"Processing (Train: {y1_train}-{y2_train}, Eval: {y1_eval}-{y2_eval}, IR={IR})...", log_file)

            # Load train data
            train_data_pandas = pd.read_parquet(train_file) 
            all_input_train = train_data_pandas.values
            train_data_pandas = pd.DataFrame()  # free memory

            # Load eval data
            eval_data_pandas = pd.read_parquet(eval_file)
            all_input_eval = eval_data_pandas.values
            eval_data_pandas = pd.DataFrame()  # free memory

            eval_feature_dataset = all_input_eval[:, 3:]  # selecting features f0,...
            eval_solution_dataset = all_input_eval[:, 2]
            all_negative_solution_eval = np.all(eval_solution_dataset == 0)

            np.random.shuffle(all_input_train)
            input_data = all_input_train[:, 3:]  # selecting features
            supervised_solution = all_input_train[:, 2]  # solutions

            train_test_size = [0.85, 0.15]
            idx_train = int(len(input_data) * train_test_size[0])
            input_data_train = input_data[:idx_train]
            train_solution = supervised_solution[:idx_train]

            input_data_test = input_data[idx_train:]
            test_solution = supervised_solution[idx_train:] 

            data_train = torch.tensor(input_data_train, dtype=torch.float32).to(device)
            solution_train = torch.tensor(train_solution, dtype=torch.float32).to(device)

            data_test = torch.tensor(input_data_test, dtype=torch.float32).to(device)
            solution_test = torch.tensor(test_solution, dtype=torch.float32).to(device)

            input_size = data_train.shape[1]
            output_size = 1

            model_semnet = ff_network(input_size, hidden_size, output_size).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model_semnet.parameters(), lr=lr_enc)

            size_of_loss_check = 10000
            # Initialize variables for early stopping
            best_test_loss = float('inf')
            best_epoch = 0

            train_loss_total = []
            test_loss_total = []

            start_time = time.time()

            num_epochs = 5000000
            print_cluster("start training....", log_file)
            for epoch in range(num_epochs):
                model_semnet.train()
                # Randomly select 'batch_size' samples from data_train
                indices = np.random.choice(len(data_train), batch_size, replace=False)
                batch_data = data_train[indices]
                batch_solution = solution_train[indices]

                # Forward pass
                optimizer.zero_grad()
                predictions = model_semnet(batch_data).squeeze()
                real_loss = criterion(predictions, batch_solution)
                loss = torch.clamp(real_loss, min=0., max=50000.).double()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    model_semnet.eval()
                    # Evaluate on a subset of the training data
                    train_predictions = model_semnet(data_train[:size_of_loss_check]).squeeze()
                    train_loss = criterion(train_predictions, solution_train[:size_of_loss_check]).item()
                    # Evaluate on a subset of the test data
                    test_predictions = model_semnet(data_test[:size_of_loss_check]).squeeze()
                    test_loss = criterion(test_predictions, solution_test[:size_of_loss_check]).item()

                train_loss_total.append(train_loss)
                test_loss_total.append(test_loss)

                # Calculate epochs since last best test loss
                epochs_since_best = epoch - best_epoch

                # Print progress
                elapsed_time = time.time() - start_time
                print_cluster(f'epoch {epoch}: Train Loss = {train_loss:.5f}, Test Loss = {test_loss:.5f}, Time = {elapsed_time:.5f}s, ESC: {epochs_since_best}/{patience}', log_file)
                start_time = time.time()
                    
                # Check if current test loss is the best so far
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_epoch = epoch
                    # Save the model when a new best is found
                    did_work=False
                    while did_work==False:
                        try:
                            net_file = os.path.join(neuralNet_folder, f"fcNN_netNet_full_trained_{file_name_prefix}.pt")
                            torch.save(model_semnet, net_file)
                            torch.save(model_semnet.state_dict(), net_file.replace("fcNN_netNet_full", "fcNN_netNet_state"))
                            did_work=True
                        except:
                            time.sleep(0.1)
                # Early stopping: if no improvement in 'patience' epochs, stop training
                if epoch - best_epoch > patience:
                    print_cluster(f'Early stopping triggered at epoch {epoch}. Best test loss {best_test_loss:.5f} was not improved for {patience} epochs.', log_file)
                    break

            print_cluster("finish training....", log_file)

            # Load the best performing model
            net_file = os.path.join(neuralNet_folder, f"fcNN_netNet_full_trained_{file_name_prefix}.pt")
            model_semnet = torch.load(net_file, map_location=device)
            model_semnet.eval()

            save_loss_file = os.path.join(plot_folder, f"fcNN_loss_curve_{file_name_prefix}.png")
            plot_loss_curve(train_loss_total, test_loss_total, save_loss_file, label=f"{y1_train}-{y2_train}")

            print_cluster("start evaluation for train, test and eval if possible....", log_file)
            eval_batch_size = 50000
            train_predictions, train_labels = get_predictions(model_semnet, data_train, solution_train, eval_batch_size, log_file=log_file)
            save_train_auc_file = os.path.join(plot_folder, f"fcNN_train_auc_curve_{file_name_prefix}.png")
            curr_auc=plot_auc_roc(train_labels, train_predictions, save_train_auc_file, label="Train")
            print_cluster(f"Train AUC: {curr_auc}", log_file)

            test_predictions, test_labels = get_predictions(model_semnet, data_test, solution_test, eval_batch_size, log_file=log_file)
            save_test_auc_file = os.path.join(plot_folder, f"fcNN_test_auc_curve_{file_name_prefix}.png")
            curr_auc=plot_auc_roc(test_labels, test_predictions, save_test_auc_file, label="Test")
            print_cluster(f"Test AUC: {curr_auc}", log_file)
            print_cluster("finish auc plot for train, test...", log_file)

            if not all_negative_solution_eval: # contain positive cases
                data_eval = torch.tensor(eval_feature_dataset, dtype=torch.float32).to(device)
                solution_eval = torch.tensor(eval_solution_dataset, dtype=torch.float32).to(device)
                eval_predictions, eval_labels = get_predictions(model_semnet, data_eval, solution_eval, eval_batch_size, log_file=log_file)
                save_eval_auc_file = os.path.join(plot_folder, f"fcNN_eval_auc_curve_{file_name_prefix}.png")
                curr_auc=plot_auc_roc(eval_labels, eval_predictions, save_eval_auc_file, label="Eval")
                print_cluster(f"Eval AUC: {curr_auc}", log_file)                

            print_cluster("finish all.....", log_file)
