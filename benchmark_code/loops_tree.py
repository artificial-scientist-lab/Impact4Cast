import os
import pickle
import gzip
import copy
import random, time
import numpy as np
import pandas as pd
from datetime import datetime, date
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for cluster
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import joblib

def print_cluster(print_str, log_file):
    print(print_str)
    with open(log_file, "a") as logfile:
        logfile.write(print_str + "\n")

def get_predictions(model, data, solution, eval_batch_size, log_file):
    # model here is a RandomForestClassifier
    # We'll do the predictions in batches if needed
    data_batches = []
    n = data.shape[0]
    idx = 0
    while idx < n:
        data_batches.append(data[idx:idx+eval_batch_size])
        idx += eval_batch_size

    all_predictions = []
    start_time = time.time()
    for i, batch in enumerate(data_batches, start=1):
        batch_start_time = time.time()
        # For random forest, predict_proba returns probabilities for each class
        # We assume solution is binary {0,1}, so we take probability of class 1
        batch_preds = model.predict_proba(batch)[:, 1]
        all_predictions.append(batch_preds)
        batch_time = time.time() - batch_start_time
        print_cluster(f"Processed batch {i}/{len(data_batches)} in {batch_time:.2f} seconds", log_file)

    predictions = np.concatenate(all_predictions)
    true_labels = solution
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

# Dummy variables for same file format of the log files
batch_size = 2048 # Dummy variables for same file format of the log files
lr_enc = 1e-4     # Dummy variables for same file format of the log files
hidden_size = 600 # Dummy variables for same file format of the log files
patience = 500    # Dummy variables for same file format of the log files
# Dummy variables for same file format of the log files

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
np.random.seed(rnd_seed)

train_year_spans = [2, 3, 4]   # (y2_train - y1_train) 
eval_year_spans = [1, 2, 3, 4, 5]  # (y2_eval - y1_eval)
IR_list = [10, 50] 
fixed_y2_eval = 2022


for t_span in train_year_spans:
    for e_span in eval_year_spans:
        # Compute eval years
        y2_eval = fixed_y2_eval
        y1_eval = y2_eval - e_span

        # Compute train years
        # given: y2_train = y1_eval
        y2_train = y1_eval
        y1_train = y2_train - t_span

        for IR in IR_list:
            # Construct a file name prefix for all files
            file_name_prefix = f"{y1_train}_{y2_train}_{hidden_size}_{lr_enc}_{batch_size}_{patience}_{IR}"
            log_file = f'log_tree_{file_name_prefix}.txt'
            
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

            eval_feature_dataset = all_input_eval[:, 3:]  # selecting features
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

            # Initialize the Random Forest classifier
            print_cluster("Finished shuffling and splitting training data...", log_file)
            
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=25,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=rnd_seed,
                verbose=1,
                class_weight='balanced'
            )

            print_cluster("Training the Random Forest classifier...", log_file)
            start_time = time.time()
            clf.fit(input_data_train, train_solution)
            end_time = time.time()
            print_cluster(f"Training completed in {end_time - start_time:.2f} seconds.", log_file)

            # We mimic the logic of plotting loss curves by calculating MSE on a subset
            size_of_loss_check = 10000
            # Compute "loss" as MSE of predictions vs solutions, just for plotting
            train_pred_for_loss = clf.predict_proba(input_data_train[:size_of_loss_check])[:,1]
            train_loss = mean_squared_error(train_solution[:size_of_loss_check], train_pred_for_loss)

            test_pred_for_loss = clf.predict_proba(input_data_test[:size_of_loss_check])[:,1]
            test_loss = mean_squared_error(test_solution[:size_of_loss_check], test_pred_for_loss)

            train_loss_total = [train_loss]
            test_loss_total = [test_loss]

            # Save the trained model (replace fcNN by tree in filenames)
            net_file = os.path.join(neuralNet_folder, f"tree_netNet_full_trained_{file_name_prefix}.pkl")
            joblib.dump(clf, net_file)

            # Plot loss curve (will be trivial, just one point)
            save_loss_file = os.path.join(plot_folder, f"tree_loss_curve_{file_name_prefix}.png")
            plot_loss_curve(train_loss_total, test_loss_total, save_loss_file, label=f"{y1_train}-{y2_train}")

            print_cluster("start evaluation for train, test and eval if possible....", log_file)
            eval_batch_size = 50000

            # Load model again to mimic original code
            clf = joblib.load(net_file)

            # Get predictions for training set
            train_predictions, train_labels = get_predictions(clf, input_data_train, train_solution, eval_batch_size, log_file=log_file)
            save_train_auc_file = os.path.join(plot_folder, f"tree_train_auc_curve_{file_name_prefix}.png")
            curr_auc = plot_auc_roc(train_labels, train_predictions, save_train_auc_file, label="Train")
            print_cluster(f"Train AUC: {curr_auc}", log_file)

            # Get predictions for test set
            test_predictions, test_labels = get_predictions(clf, input_data_test, test_solution, eval_batch_size, log_file=log_file)
            save_test_auc_file = os.path.join(plot_folder, f"tree_test_auc_curve_{file_name_prefix}.png")
            curr_auc = plot_auc_roc(test_labels, test_predictions, save_test_auc_file, label="Test")
            print_cluster(f"Test AUC: {curr_auc}", log_file)
            print_cluster("finish auc plot for train, test...", log_file)

            if not all_negative_solution_eval: # contain positive cases
                eval_predictions, eval_labels = get_predictions(clf, eval_feature_dataset, eval_solution_dataset, eval_batch_size, log_file=log_file)
                save_eval_auc_file = os.path.join(plot_folder, f"tree_eval_auc_curve_{file_name_prefix}.png")
                curr_auc = plot_auc_roc(eval_labels, eval_predictions, save_eval_auc_file, label="Eval")
                print_cluster(f"Eval AUC: {curr_auc}", log_file)                

            print_cluster("finish all.....", log_file)
